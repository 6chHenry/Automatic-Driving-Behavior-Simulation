# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

'''
    利用traj-clustering.py生成的动作词汇表,为轨迹离散化
'''
import os
import pickle
from typing import Dict, Tuple

import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.data import HeteroData

from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
)


class TokenProcessor(torch.nn.Module):

    def __init__(
        self,
        map_token_file: str,
        agent_token_file: str,
        map_token_sampling: DictConfig,
        agent_token_sampling: DictConfig,
    ) -> None:
        super(TokenProcessor, self).__init__()
        self.map_token_sampling = map_token_sampling
        self.agent_token_sampling = agent_token_sampling
        self.shift = 5

        module_dir = os.path.dirname(__file__)
        # 加载预训练词汇表
        self.init_agent_token(os.path.join(module_dir, agent_token_file)) # 加载运动token
        self.init_map_token(os.path.join(module_dir, map_token_file))     # 加载地图token
        self.n_token_agent = self.agent_token_all_veh.shape[0]   	# 运动 token 词汇表大小

    @torch.no_grad()   # 不求梯度
    def forward(self, data: HeteroData) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        tokenized_map = self.tokenize_map(data)
        tokenized_agent = self.tokenize_agent(data)
        return tokenized_map, tokenized_agent

    def init_map_token(self, map_token_traj_path, argmin_sample_len=3) -> None:
        map_token_traj = pickle.load(open(map_token_traj_path, "rb"))["traj_src"]
        # [n_tokens , sequence_length or num_steps, 2]    =    [n_tokens, 11, 2]

        # 均匀生成armin_sample_len个采样点
        indices = torch.linspace(
            0, map_token_traj.shape[1] - 1, steps=argmin_sample_len
        ).long()

        self.register_buffer(
            "map_token_traj_src",
            torch.tensor(map_token_traj, dtype=torch.float32).flatten(1, 2),
            persistent=False,
        )  # [n_token, 11*2]    时空维度合并
        # register_buffer 创建的对象不参加梯度计算,优化器更新

        self.register_buffer(
            "map_token_sample_pt",
            torch.tensor(map_token_traj[:, indices], dtype=torch.float32).unsqueeze(0),
            persistent=False,
        )  # [1, n_token, 3, 2]  添加批次维度

    def init_agent_token(self, agent_token_path) -> None:
        agent_token_data = pickle.load(open(agent_token_path, "rb"))
        for k, v in agent_token_data["token_all"].items():
            v = torch.tensor(v, dtype=torch.float32)
            # [n_token, 6, 4, 2], countour, 10 hz
            self.register_buffer(f"agent_token_all_{k}", v, persistent=False)

    # n_token是预定义地图特征原型的数量(也就是Vocabulary Size), n_pl是场景中需要处理的地图元素实例总数(也就是batch_size)
    def tokenize_map(self, data: HeteroData) -> Dict[str, Tensor]:
        traj_pos = data["map_save"]["traj_pos"]  # [n_pl, 3, 2]   n_pl是地图元素数量, 3是提取的时间点数量, 2是(x,y)
        traj_theta = data["map_save"]["traj_theta"]  # [n_pl] 初始朝向

        traj_pos_local, _ = transform_to_local(
            pos_global=traj_pos,  # [n_pl, 3, 2]
            head_global=None,  # [n_pl, 1]
            pos_now=traj_pos[:, 0],  # [n_pl, 2]
            head_now=traj_theta,  # [n_pl]
        )

        # [1, n_token, 3, 2] - [n_pl, 1, 3, 2]   broadcast
        dist = torch.sum(
            (self.map_token_sample_pt - traj_pos_local.unsqueeze(1)) ** 2,
            dim=(-2, -1),
        )  # [n_pl, n_token]

        if self.training and (self.map_token_sampling.num_k > 1):   # 训练模式, 从K个中选
            topk_dists, topk_indices = torch.topk(
                dist,
                self.map_token_sampling.num_k,
                dim=-1,
                largest=False,
                sorted=False,
            )  # [n_pl, K]   每个地图元素对应的前K小距离 及 对应的下标

            topk_logits = (-1e-6 - topk_dists) / self.map_token_sampling.temp
            _samples = Categorical(logits=topk_logits).sample()  # [n_pl] in K   softmax
            token_idx = topk_indices[torch.arange(len(_samples)), _samples].contiguous()
            # token_idx = topk_indices[row_indices, col_indices]= [topk_indices[0, k_0], topk_indices[1, k_1], ...]

        else:  # 推理模式, 选最佳的那一个
            token_idx = torch.argmin(dist, dim=-1)

        tokenized_map = {
            "position": traj_pos[:, 0].contiguous(),  # [n_pl, 2]  起点位置
            "orientation": traj_theta,  # [n_pl]  初始朝向
            "token_idx": token_idx,  # [n_pl]
            "token_traj_src": self.map_token_traj_src,  # [n_token, 11*2]
            "type": data["pt_token"]["type"].long(),  # [n_pl]  元素类型
            "pl_type": data["pt_token"]["pl_type"].long(),  # [n_pl]  车道类型
            "light_type": data["pt_token"]["light_type"].long(),  # [n_pl]  交通灯状态
            "batch": data["pt_token"]["batch"],  # [n_pl]
        }
        return tokenized_map

    def tokenize_agent(self, data: HeteroData) -> Dict[str, Tensor]:
        """
        Args: data["agent"]: Dict
            "valid_mask": [n_agent, n_step], bool
            "role": [n_agent, 3], bool
            "id": [n_agent], int64
            "type": [n_agent], uint8
            "position": [n_agent, n_step, 3], float32
            "heading": [n_agent, n_step], float32
            "velocity": [n_agent, n_step, 2], float32
            "shape": [n_agent, 3], float32
        """
        # ! collate width/length, traj tokens for current batch
        agent_shape, token_traj_all, token_traj = self._get_agent_shape_and_token_traj(
            data["agent"]["type"]
        )

        # ! get raw trajectory data
        valid = data["agent"]["valid_mask"]  # [n_agent, n_step]
        heading = data["agent"]["heading"]  # [n_agent, n_step]
        pos = data["agent"]["position"][..., :2].contiguous()  # [n_agent, n_step, 2]
        vel = data["agent"]["velocity"]  # [n_agent, n_step, 2]

        # ! agent, specifically vehicle's heading can be 180 degree off. We fix it here.
        heading = self._clean_heading(valid, heading)

        # ! extrapolate to previous 5th step. 补全前5步
        valid, pos, heading, vel = self._extrapolate_agent_to_prev_token_step(
            valid, pos, heading, vel
        )

        # ! prepare output dict
        tokenized_agent = {
            "num_graphs": data.num_graphs,
            "type": data["agent"]["type"],
            "shape": data["agent"]["shape"],
            "ego_mask": data["agent"]["role"][:, 0],  # [n_agent]  自我标识
            "token_agent_shape": agent_shape,  # [n_agent, 2]
            "batch": data["agent"]["batch"],
            "token_traj_all": token_traj_all,  # [n_agent, n_token, 6, 4, 2]
            "token_traj": token_traj,  # [n_agent, n_token, 4, 2]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": pos[:, self.shift :: self.shift],  # [n_agent, n_step=18, 2]
            "gt_head_raw": heading[:, self.shift :: self.shift],  # [n_agent, n_step=18]
            "gt_valid_raw": valid[:, self.shift :: self.shift],  # [n_agent, n_step=18]
        }

        # [n_token, 8]
        for k in ["veh", "ped", "cyc"]:
            tokenized_agent[f"trajectory_token_{k}"] = getattr(self, f"agent_token_all_{k}" )[:, -1].flatten(1, 2)  

        # ! match token for each agent
        if not self.training:
            # [n_agent]
            tokenized_agent["gt_z_raw"] = data["agent"]["position"][:, 10, 2]   # 第10个时间步的高度

        token_dict = self._match_agent_token(
            valid=valid,
            pos=pos,
            heading=heading,
            agent_shape=agent_shape,
            token_traj=token_traj,
        )
        tokenized_agent.update(token_dict)  # 将两个字典合并 
        return tokenized_agent

    def _match_agent_token(
        self,
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        agent_shape: Tensor,  # [n_agent, 2]
        token_traj: Tensor,  # [n_agent, n_token, 4, 2] 
    ) -> Dict[str, Tensor]:
        """n_step_token=n_step//5
        n_step_token=18 for train with BC.
        n_step_token=2 for val/test and train with closed-loop rollout.
        Returns: Dict
            # ! action that goes from [(0->5), (5->10), ..., (85->90)]
            "valid_mask": [n_agent, n_step_token]
            "gt_idx": [n_agent, n_step_token]
            # ! at step [5, 10, 15, ..., 90]
            "gt_pos": [n_agent, n_step_token, 2]
            "gt_heading": [n_agent, n_step_token]
            # ! noisy sampling for training data augmentation
            "sampled_idx": [n_agent, n_step_token]
            "sampled_pos": [n_agent, n_step_token, 2]
            "sampled_heading": [n_agent, n_step_token]
        """
        num_k = self.agent_token_sampling.num_k if self.training else 1
        n_agent, n_step = valid.shape
        range_a = torch.arange(n_agent)

        prev_pos, prev_head = pos[:, 0], heading[:, 0]  # [n_agent, 2], [n_agent]
        prev_pos_sample, prev_head_sample = pos[:, 0], heading[:, 0]

        out_dict = {
            "valid_mask": [],
            "gt_idx": [],
            "gt_pos": [],
            "gt_heading": [],
            "sampled_idx": [],
            "sampled_pos": [],
            "sampled_heading": [],
        }

        for i in range(self.shift, n_step, self.shift):  # [5, 10, 15, ..., 90]   shift是步长
            _valid_mask = valid[:, i - self.shift] & valid[:, i]  # [n_agent]
            _invalid_mask = ~_valid_mask
            out_dict["valid_mask"].append(_valid_mask)

            #! gt_contour: [n_agent, 4, 2] in global coord
            gt_contour = cal_polygon_contour(pos[:, i], heading[:, i], agent_shape)
            gt_contour = gt_contour.unsqueeze(1)  # [n_agent, 1, 4, 2]

            # ! tokenize without sampling
            token_world_gt = transform_to_global(
                pos_local=token_traj.flatten(1, 2),  # [n_agent, n_token*4, 2]
                head_local=None,
                pos_now=prev_pos,  # [n_agent, 2]
                head_now=prev_head,  # [n_agent]
            )[0].view(*token_traj.shape)  # [n_agent, n_token, 4, 2]
            token_idx_gt = torch.argmin(
                torch.norm(token_world_gt - gt_contour, dim=-1).sum(-1), dim=-1
            )  # [n_agent]   从n_token中匹配到一个最相近的
            token_contour_gt = token_world_gt[range_a, token_idx_gt] # [n_agent, 4, 2]    高级index操作, 得到全局坐标系下的离散化轨迹

            # udpate prev_pos, prev_head
            prev_head = heading[:, i].clone()
            dxy = token_contour_gt[:, 0] - token_contour_gt[:, 3]     # 计算方向向量（首尾点连线）
            prev_head[_valid_mask] = torch.arctan2(dxy[:, 1], dxy[:, 0])[_valid_mask]
            prev_pos = pos[:, i].clone()
            prev_pos[_valid_mask] = token_contour_gt.mean(1)[_valid_mask]

            # add to output dict
            out_dict["gt_idx"].append(token_idx_gt)
            out_dict["gt_pos"].append(
                prev_pos.masked_fill(_invalid_mask.unsqueeze(1), 0)   # 无效位置填0
            )
            out_dict["gt_heading"].append(prev_head.masked_fill(_invalid_mask, 0))

            # ! tokenize from sampled rollout state
            # gt_idx是最接近真实值的, sampled_idx是根据采样规则得到的(如在Top k中选)
            if num_k == 1:  # K=1 means no sampling
                out_dict["sampled_idx"].append(out_dict["gt_idx"][-1])
                out_dict["sampled_pos"].append(out_dict["gt_pos"][-1])
                out_dict["sampled_heading"].append(out_dict["gt_heading"][-1])
            else:
                # contour: [n_agent, n_token, 4, 2], 2Hz, global coord
                token_world_sample = transform_to_global(
                    pos_local=token_traj.flatten(1, 2),  # [n_agent, n_token*4, 2]
                    head_local=None,
                    pos_now=prev_pos_sample,  # [n_agent, 2]
                    head_now=prev_head_sample,  # [n_agent]
                )[0].view(*token_traj.shape)

                dist = torch.norm(token_world_sample - gt_contour, dim=-1).mean(-1)  # dist: [n_agent, n_token]
                topk_dists, topk_indices = torch.topk(dist, num_k, dim=-1, largest=False, sorted=False)  # [n_agent, K]

                topk_logits = (-1.0 * topk_dists) / self.agent_token_sampling.temp
                _samples = Categorical(logits=topk_logits).sample()  # [n_agent] in K  (先softmax, 再从K个候选中按概率抽一个)
                token_idx_sample = topk_indices[range_a, _samples]   # [n_agent]
                token_contour_sample = token_world_sample[range_a, token_idx_sample]   # [n_agent, 4, 2]

                # udpate prev_pos_sample, prev_head_sample
                prev_head_sample = heading[:, i].clone()
                dxy = token_contour_sample[:, 0] - token_contour_sample[:, 3]
                prev_head_sample[_valid_mask] = torch.arctan2(dxy[:, 1], dxy[:, 0])[_valid_mask]
                prev_pos_sample = pos[:, i].clone()
                prev_pos_sample[_valid_mask] = token_contour_sample.mean(1)[_valid_mask]

                # add to output dict
                out_dict["sampled_idx"].append(token_idx_sample)
                out_dict["sampled_pos"].append(prev_pos_sample.masked_fill(_invalid_mask.unsqueeze(1), 0.0))
                out_dict["sampled_heading"].append(prev_head_sample.masked_fill(_invalid_mask, 0.0))

        out_dict = {k: torch.stack(v, dim=1) for k, v in out_dict.items()}   # 将列表转换为张量, 在时间维度堆叠
        return out_dict

    @staticmethod
    def _clean_heading(valid: Tensor, heading: Tensor) -> Tensor:
        valid_pairs = valid[:, :-1] & valid[:, 1:]
        for i in range(heading.shape[1] - 1):
            heading_diff = torch.abs(wrap_angle(heading[:, i] - heading[:, i + 1]))
            change_needed = (heading_diff > 1.5) & valid_pairs[:, i]
            heading[:, i + 1][change_needed] = heading[:, i][change_needed]
        return heading

    def _extrapolate_agent_to_prev_token_step(
        self,
        valid: Tensor,  # [n_agent, n_step]
        pos: Tensor,  # [n_agent, n_step, 2]
        heading: Tensor,  # [n_agent, n_step]
        vel: Tensor,  # [n_agent, n_step, 2]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # [n_agent], max will give the first True step
        first_valid_step = torch.max(valid, dim=1).indices

        for i, t in enumerate(first_valid_step):  # extrapolate to previous 5th step.
            n_step_to_extrapolate = t % self.shift
            if (t == 10) and (not valid[i, 10 - self.shift]):
                # such that at least one token is valid in the history.
                n_step_to_extrapolate = self.shift

            if n_step_to_extrapolate > 0:
                vel[i, t - n_step_to_extrapolate : t] = vel[i, t]
                valid[i, t - n_step_to_extrapolate : t] = True
                heading[i, t - n_step_to_extrapolate : t] = heading[i, t]

                for j in range(n_step_to_extrapolate):
                    pos[i, t - j - 1] = pos[i, t - j] - vel[i, t] * 0.1

        return valid, pos, heading, vel

    def _get_agent_shape_and_token_traj(
        self, agent_type: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        agent_shape: [n_agent, 2]
        token_traj_all: [n_agent, n_token, 6, 4, 2]
        token_traj: [n_agent, n_token, 4, 2]
        """
        agent_type_masks = {
            "veh": agent_type == 0,
            "ped": agent_type == 1,
            "cyc": agent_type == 2,
        }
        agent_shape = 0.0
        token_traj_all = 0.0
        for k, mask in agent_type_masks.items():
            if k == "veh":
                width = 2.0
                length = 4.8
            elif k == "cyc":
                width = 1.0
                length = 2.0
            else:
                width = 1.0
                length = 1.0
            agent_shape += torch.stack([width * mask, length * mask], dim=-1)

            token_data = getattr(self, f"agent_token_all_{k}")  # 获取预存标记 [n_token, 6, 4, 2]
            expanded_data = token_data.unsqueeze(0)             # 增加批维度 [1, n_token, 6, 4, 2]
            mask_expanded = mask[:, None, None, None, None]     # mask原始shape[n_agent], 扩展mask维度 [n_agent, 1, 1, 1, 1]
            token_traj_all += mask_expanded * expanded_data

        token_traj = token_traj_all[:, :, -1, :, :].contiguous()
        return agent_shape, token_traj_all, token_traj
               # 尺寸, 完整轨迹标记(6个时间步)[n_agent, n_token, 6, 4, 2], 末端轨迹标志(最后1个时间步)[n_agent, n_token, 4, 2]
