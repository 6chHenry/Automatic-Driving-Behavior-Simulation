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

from typing import Dict, Optional
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_cluster import radius, radius_graph
from torch_geometric.utils import dense_to_sparse, subgraph

from src.smart.layers import MLPLayer
from src.smart.layers import MLPHead
from src.smart.layers.attention_layer import AttentionLayer
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    angle_between_2d_vectors,
    sample_next_token_traj,
    transform_to_global,
    weight_init,
    wrap_angle,
)


class SMARTAgentDecoder(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,    # 地图到智能体交互半径
        a2a_radius: float,     # 智能体间交互半径
        num_freq_bands: int,
        num_layers: int,       # 注意力层数
        num_heads: int,        # 注意力头数
        head_dim: int,         # 注意力头维度
        dropout: float, 
        hist_drop_prob: float,
        n_token_agent: int,
        n_pred_steps: int = 16,  # 新增参数，multi-token预测步数
    ) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_layers = num_layers
        self.shift = 5
        self.hist_drop_prob = hist_drop_prob

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        input_dim_token = 8

        self.type_a_emb = nn.Embedding(3, hidden_dim)   # 类型嵌入
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)  # 形状嵌入

        self.x_a_emb = FourierEmbedding(  # 运动特征嵌入(position/heading)
            input_dim=input_dim_x_a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        
        # 相对位置嵌入
        self.r_t_emb = FourierEmbedding(     # 时域
            input_dim=input_dim_r_t,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_pt2a_emb = FourierEmbedding(  # map to agent
            input_dim=input_dim_r_pt2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_a2a_emb = FourierEmbedding(   # agent to agent
            input_dim=input_dim_r_a2a,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )

        # token嵌入
        self.token_emb_veh = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.token_emb_ped = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.token_emb_cyc = MLPEmbedding(
            input_dim=input_dim_token, hidden_dim=hidden_dim
        )
        self.fusion_emb = MLPEmbedding(
            input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim
        )

        # 时域注意力
        self.t_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        # map-agent交叉注意力
        self.pt2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        # agent间自注意力
        self.a2a_attn_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent
        )
        # 新增multi-token预测头
        self.multi_token_predict_head = MLPLayer(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=n_token_agent * n_pred_steps
        )
        self.n_pred_steps = n_pred_steps

        # 解码头解出x,y
        self.xy_decoder_veh = MLPHead(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=8
        )
        self.xy_decoder_ped = MLPHead(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=8
        )
        self.xy_decoder_cyc = MLPHead(
            input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=8
        )
        
        self.apply(weight_init)

    def agent_token_embedding(
        self,
        agent_token_index,  # [n_agent, n_step]  离散化后的运动token序列，每个位置的值是预定义token词汇表中的索引
        trajectory_token_veh,  # [n_token, 8]   预训练的运动token库   8 = 4 * 2
        trajectory_token_ped,  # [n_token, 8]
        trajectory_token_cyc,  # [n_token, 8]
        pos_a,  # [n_agent, n_step, 2]
        head_vector_a,  # [n_agent, n_step, 2]
        agent_type,  # [n_agent]
        agent_shape,  # [n_agent, 3]
        inference=False,
    ):
        n_agent, n_step, traj_dim = pos_a.shape
        _device = pos_a.device

        veh_mask = agent_type == 0
        ped_mask = agent_type == 1
        cyc_mask = agent_type == 2

        #  [n_token, hidden_dim]
        agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh)
        agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped)
        agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc)

        #[n_agent, n_step, hidden_dim]
        agent_token_emb = torch.zeros(
            (n_agent, n_step, self.hidden_dim), device=_device, dtype=pos_a.dtype
        )
        agent_token_emb[veh_mask] = agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = agent_token_emb_cyc[agent_token_index[cyc_mask]]

        
        motion_vector_a = torch.cat(
            [
                pos_a.new_zeros(agent_token_index.shape[0], 1, traj_dim),
                pos_a[:, 1:] - pos_a[:, :-1],
            ],
            dim=1,
        )  # [n_agent, n_step, 2] 瞬时位移(速度向量)

        feature_a = torch.stack(
            [
                torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]
                ),
            ],
            dim=-1,
        )  # [n_agent, n_step, 2]   两个特征:1.速度大小  2.速度向量与朝向的夹角(弧度制)

        categorical_embs = [
            self.type_a_emb(agent_type.long()),
            self.shape_emb(agent_shape),
        ]  # List of len=2, shape [n_agent, hidden_dim]   类型嵌入

        x_a = self.x_a_emb(
            continuous_inputs=feature_a.view(-1, feature_a.size(-1)),   # [n_agent*n_step, 2]
            categorical_embs=[
                v.repeat_interleave(repeats=n_step, dim=0) for v in categorical_embs
            ],   # [n_agent*n_step, hidden_dim]
        )  # [n_agent*n_step, hidden_dim]
        x_a = x_a.view(-1, n_step, self.hidden_dim)  # [n_agent, n_step, hidden_dim]   恢复形状

        '''agent_token_emb是来自离散动作词汇表的嵌入, x_a是连续特征和分类特征的嵌入'''
        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)   # [n_agent, n_step, 2*hidden_dim]
        feat_a = self.fusion_emb(feat_a)   # [n_agent, n_step, hidden_dim]

        xy_pred_veh = self.xy_decoder_veh(agent_token_emb_veh)  # [n_token, 8]
        xy_pred_ped = self.xy_decoder_ped(agent_token_emb_ped)  # [n_token, 8]
        xy_pred_cyc = self.xy_decoder_cyc(agent_token_emb_cyc)  # [n_token, 8]
     
        l2_loss = 0.0
        # 计算L2loss
        if not inference:  # 训练时计算损失
            l2_loss_veh = (torch.nn.functional.mse_loss(
                xy_pred_veh, # [n_token, 8]
                trajectory_token_veh, # [n_token, 8]
                reduction='none'
            ).sum(dim=-1).mean())
            l2_loss_ped = (torch.nn.functional.mse_loss(
                xy_pred_ped, # [n_token, 8]
                trajectory_token_ped, # [n_token, 8]
                reduction='none'
            ).sum(dim=-1).mean())
            l2_loss_cyc = (torch.nn.functional.mse_loss(
                xy_pred_cyc, # [n_token, 8]
                trajectory_token_cyc, # [n_token, 8]
                reduction='none'
            ).sum(dim=-1).mean())
            l2_loss = (l2_loss_veh + l2_loss_ped + l2_loss_cyc)/3.0

        if inference:
            return (
                feat_a,  # [n_agent, n_step, hidden_dim]
                l2_loss,
                agent_token_emb,  # [n_agent, n_step, hidden_dim]
                agent_token_emb_veh,  # [n_agent, hidden_dim]
                agent_token_emb_ped,  # [n_agent, hidden_dim]
                agent_token_emb_cyc,  # [n_agent, hidden_dim]
                veh_mask,  # [n_agent]
                ped_mask,  # [n_agent]
                cyc_mask,  # [n_agent]
                categorical_embs,  # List of len=2, shape [n_agent, hidden_dim]
            )
        else:
            return feat_a, l2_loss  # [n_agent, n_step, hidden_dim]

    def build_temporal_edge(
        self,
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2],
        mask,  # [n_agent, n_step]
        inference_mask=None,  # [n_agent, n_step]
    ):
        pos_t = pos_a.flatten(0, 1)
        head_t = head_a.flatten(0, 1)
        head_vector_t = head_vector_a.flatten(0, 1)

        if self.hist_drop_prob > 0 and self.training:
            _mask_keep = torch.bernoulli(
                torch.ones_like(mask) * (1 - self.hist_drop_prob)
            ).bool()
            mask = mask & _mask_keep

        if inference_mask is not None:
            mask_t = mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)

        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift
        ]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_pos_t = rel_pos_t[:, :2]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [
                torch.norm(rel_pos_t, p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t
                ),
                rel_head_t,
                edge_index_t[0] - edge_index_t[1],
            ],
            dim=-1,
        )
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(   # agent2agent
        self,
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2]
        batch_s,  # [n_agent*n_step]
        mask,  # [n_agent, n_step]
    ):
        mask = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        edge_index_a2a = radius_graph(
            x=pos_s[:, :2],
            r=self.a2a_radius,
            batch=batch_s,
            loop=False,
            max_num_neighbors=300,
        )
        edge_index_a2a = subgraph(subset=mask, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [
                torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_a2a[1]],
                    nbr_vector=rel_pos_a2a[:, :2],
                ),
                rel_head_a2a,
            ],
            dim=-1,
        )
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def build_map2agent_edge(
        self,
        pos_pl,  # [n_pl, 2]
        orient_pl,  # [n_pl]
        pos_a,  # [n_agent, n_step, 2]
        head_a,  # [n_agent, n_step]
        head_vector_a,  # [n_agent, n_step, 2]
        mask,  # [n_agent, n_step]
        batch_s,  # [n_agent*n_step]
        batch_pl,  # [n_pl*n_step]
    ):
        n_step = pos_a.shape[1]
        mask_pl2a = mask.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).flatten(0, 1)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl = pos_pl.repeat(n_step, 1)
        orient_pl = orient_pl.repeat(n_step)
        edge_index_pl2a = radius(
            x=pos_s[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2a_radius,
            batch_x=batch_s,
            batch_y=batch_pl,
            max_num_neighbors=300,
        )
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
        )
        r_pl2a = torch.stack(
            [
                torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(
                    ctr_vector=head_vector_s[edge_index_pl2a[1]],
                    nbr_vector=rel_pos_pl2a[:, :2],
                ),
                rel_orient_pl2a,
            ],
            dim=-1,
        )
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        return edge_index_pl2a, r_pl2a

    def forward(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:  # 修改类型注解为 Any 以兼容 None
        mask = tokenized_agent["valid_mask"]
        pos_a = tokenized_agent["sampled_pos"]
        head_a = tokenized_agent["sampled_heading"]
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        n_agent, n_step = head_a.shape

        if self.training:
            # 训练模式：只接收2个返回值
            feat_a, l2_loss = self.agent_token_embedding(
                agent_token_index=tokenized_agent["sampled_idx"],
                trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
                trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
                trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
                pos_a=pos_a,
                head_vector_a=head_vector_a,
                agent_type=tokenized_agent["type"],
                agent_shape=tokenized_agent["shape"],
                inference=False
            )
        else:
            # 推理/验证模式：接收多个值但只取前2个
            outputs = self.agent_token_embedding(
                agent_token_index=tokenized_agent["sampled_idx"],
                trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
                trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
                trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
                pos_a=pos_a,
                head_vector_a=head_vector_a,
                agent_type=tokenized_agent["type"],
                agent_shape=tokenized_agent["shape"],
                inference=True
            )
            feat_a = outputs[0]
            l2_loss = outputs[1]

        '''
        写一个MLP实现 [n_agent, n_step, hidden_dim]   -> [n_agent, n_step, 2 + 2]
        loss = L2loss(pos_a_new - pos_a) + L2loss(head_vector_a_new - head_vecotor_a)
        梯度下降更新这个MLP的weights和原来Embedding的weights ?
        '''

        # ! build temporal, interaction and map2agent edges
        edge_index_t, r_t = self.build_temporal_edge(
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask=mask,  # [n_agent, n_step]
        )  
        # edge_index_t: [2, n_edge_t], r_t: [n_edge_t, hidden_dim]
        '''
        edge_index = torch.tensor([[0, 1, 2],  # 源节点
                          [1, 2, 0]], dtype=torch.long)  # 目标节点
        表示3条边 : 0→1, 1→2, 2→0
        r_t表示每条边的几何/时空特征
        '''

        batch_s = torch.cat(
            [
                tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )  # [n_agent*n_step]
        batch_pl = torch.cat(
            [
                map_feature["batch"] + tokenized_agent["num_graphs"] * t
                for t in range(n_step)
            ],
            dim=0,
        )  # [n_pl*n_step]

        edge_index_a2a, r_a2a = self.build_interaction_edge(
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            batch_s=batch_s,  # [n_agent*n_step]
            mask=mask,  # [n_agent, n_step]
        )  # edge_index_a2a: [2, n_edge_a2a], r_a2a: [n_edge_a2a, hidden_dim]

        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            pos_pl=map_feature["position"],  # [n_pl, 2]
            orient_pl=map_feature["orientation"],  # [n_pl]
            pos_a=pos_a,  # [n_agent, n_step, 2]
            head_a=head_a,  # [n_agent, n_step]
            head_vector_a=head_vector_a,  # [n_agent, n_step, 2]
            mask=mask,  # [n_agent, n_step]
            batch_s=batch_s,  # [n_agent*n_step]
            batch_pl=batch_pl,  # [n_pl*n_step]
        )

        # ! attention layers
        # [n_step*n_pl, hidden_dim]
        feat_map = (
            map_feature["pt_token"].unsqueeze(0).expand(n_step, -1, -1).flatten(0, 1)
        )

        for i in range(self.num_layers):
            feat_a = feat_a.flatten(0, 1)  # [n_agent*n_step, hidden_dim]
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            # [n_step*n_agent, hidden_dim]
            feat_a = feat_a.view(n_agent, n_step, -1).transpose(0, 1).flatten(0, 1)
            feat_a = self.pt2a_attn_layers[i](
                (feat_map, feat_a), r_pl2a, edge_index_pl2a
            )
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.view(n_step, n_agent, -1).transpose(0, 1)   # 交换前两个维度,最终得到[n_agent, n_steps, hidden_dim]

        # ! final mlp to get outputs
        next_token_logits = self.token_predict_head(feat_a)  # [n_agent, n_steps, n_token_agent]
        # 新增multi-token预测
        multi_token_logits = self.multi_token_predict_head(feat_a)  # [n_agent, n_steps, n_token_agent * n_pred_steps]
        n_agent, n_steps, _ = multi_token_logits.shape
        multi_token_logits = multi_token_logits.view(n_agent, n_steps, self.n_pred_steps, -1)  # [n_agent, n_steps, n_pred_steps, n_token_agent]
        return {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": next_token_logits[:, 1:-1],  # [n_agent, 16, n_token]
            "next_token_valid": tokenized_agent["valid_mask"][:, 1:-1],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": tokenized_agent["sampled_pos"],  # [n_agent, 18, 2]
            "pred_head": tokenized_agent["sampled_heading"],  # [n_agent, 18]
            "pred_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            "l2_loss": l2_loss,  # L2损失
            # 新增multi-token预测输出
            "multi_token_logits": multi_token_logits[:, 1:-1],  # [n_agent, 16, n_pred_steps, n_token_agent]
        }   

    def inference(
        self,
        tokenized_agent: Dict[str, torch.Tensor],
        map_feature: Dict[str, torch.Tensor],
        sampling_scheme: DictConfig,
    ) -> Dict[str, Any]:  # 修改类型注解为 Any 以兼容 None
        n_agent = tokenized_agent["valid_mask"].shape[0]
        n_step_future_10hz = self.num_future_steps  # 80
        n_step_future_2hz = n_step_future_10hz // self.shift  # 16
        step_current_10hz = self.num_historical_steps - 1  # 10
        step_current_2hz = step_current_10hz // self.shift  # 2

        pos_a = tokenized_agent["gt_pos"][:, :step_current_2hz].clone()
        head_a = tokenized_agent["gt_heading"][:, :step_current_2hz].clone()
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        pred_idx = tokenized_agent["gt_idx"].clone()
        agent_token_embedding_out = self.agent_token_embedding(
            agent_token_index=tokenized_agent["gt_idx"][:, :step_current_2hz],
            trajectory_token_veh=tokenized_agent["trajectory_token_veh"],
            trajectory_token_ped=tokenized_agent["trajectory_token_ped"],
            trajectory_token_cyc=tokenized_agent["trajectory_token_cyc"],
            pos_a=pos_a,
            head_vector_a=head_vector_a,
            agent_type=tokenized_agent["type"],
            agent_shape=tokenized_agent["shape"],
            inference=True,
        )
        (
            feat_a,
            l2_loss,
            agent_token_emb,
            agent_token_emb_veh,
            agent_token_emb_ped,
            agent_token_emb_cyc,
            veh_mask,
            ped_mask,
            cyc_mask,
            categorical_embs,
        ) = agent_token_embedding_out

        if not self.training:
            pred_traj_10hz = torch.zeros(
                [n_agent, n_step_future_10hz, 2], dtype=pos_a.dtype, device=pos_a.device
            )
            pred_head_10hz = torch.zeros(
                [n_agent, n_step_future_10hz], dtype=pos_a.dtype, device=pos_a.device
            )

        pred_valid = tokenized_agent["valid_mask"].clone()
        next_token_logits_list = []
        next_token_action_list = []
        feat_a_t_dict = {}
        for t in range(n_step_future_2hz):  # 0 -> 15
            t_now = step_current_2hz - 1 + t  # 1 -> 16
            n_step = t_now + 1  # 2 -> 17

            if t == 0:  # init
                hist_step = step_current_2hz
                batch_s = torch.cat(
                    [
                        tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
                        for t in range(hist_step)
                    ],
                    dim=0,
                )
                batch_pl = torch.cat(
                    [
                        map_feature["batch"] + tokenized_agent["num_graphs"] * t
                        for t in range(hist_step)
                    ],
                    dim=0,
                )
                inference_mask = pred_valid[:, :n_step]
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a,
                    head_a=head_a,
                    head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                )
            else:
                hist_step = 1
                batch_s = tokenized_agent["batch"]
                batch_pl = map_feature["batch"]
                inference_mask = pred_valid[:, :n_step].clone()
                inference_mask[:, :-1] = False
                edge_index_t, r_t = self.build_temporal_edge(
                    pos_a=pos_a,
                    head_a=head_a,
                    head_vector_a=head_vector_a,
                    mask=pred_valid[:, :n_step],
                    inference_mask=inference_mask,
                )
                edge_index_t[1] = (edge_index_t[1] + 1) // n_step - 1

            # In the inference stage, we only infer the current stage for recurrent
            edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
                pos_pl=map_feature["position"],  # [n_pl, 2]
                orient_pl=map_feature["orientation"],  # [n_pl]
                pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                head_vector_a=head_vector_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
                batch_s=batch_s,  # [n_agent*hist_step]
                batch_pl=batch_pl,  # [n_pl*hist_step]
            )
            edge_index_a2a, r_a2a = self.build_interaction_edge(
                pos_a=pos_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                head_a=head_a[:, -hist_step:],  # [n_agent, hist_step]
                head_vector_a=head_vector_a[:, -hist_step:],  # [n_agent, hist_step, 2]
                batch_s=batch_s,  # [n_agent*hist_step]
                mask=inference_mask[:, -hist_step:],  # [n_agent, hist_step]
            )

            # ! attention layers
            for i in range(self.num_layers):
                # [n_agent, n_step, hidden_dim]
                _feat_temporal = feat_a if i == 0 else feat_a_t_dict[i]

                if t == 0:  # init, process hist_step together
                    _feat_temporal = self.t_attn_layers[i](
                        _feat_temporal.flatten(0, 1), r_t, edge_index_t
                    ).view(n_agent, n_step, -1)
                    _feat_temporal = _feat_temporal.transpose(0, 1).flatten(0, 1)

                    # [hist_step*n_pl, hidden_dim]
                    _feat_map = (
                        map_feature["pt_token"]
                        .unsqueeze(0)
                        .expand(hist_step, -1, -1)
                        .flatten(0, 1)
                    )

                    _feat_temporal = self.pt2a_attn_layers[i](
                        (_feat_map, _feat_temporal), r_pl2a, edge_index_pl2a
                    )
                    _feat_temporal = self.a2a_attn_layers[i](
                        _feat_temporal, r_a2a, edge_index_a2a
                    )
                    _feat_temporal = _feat_temporal.view(n_step, n_agent, -1).transpose(
                        0, 1
                    )
                    feat_a_now = _feat_temporal[:, -1]  # [n_agent, hidden_dim]

                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = _feat_temporal

                else:  # process one step
                    feat_a_now = self.t_attn_layers[i](
                        (_feat_temporal.flatten(0, 1), _feat_temporal[:, -1]),
                        r_t,
                        edge_index_t,
                    )
                    # * give same results as below, but more efficient
                    # feat_a_now = self.t_attn_layers[i](
                    #     _feat_temporal.flatten(0, 1), r_t, edge_index_t
                    # ).view(n_agent, n_step, -1)[:, -1]

                    feat_a_now = self.pt2a_attn_layers[i](
                        (map_feature["pt_token"], feat_a_now), r_pl2a, edge_index_pl2a
                    )
                    feat_a_now = self.a2a_attn_layers[i](
                        feat_a_now, r_a2a, edge_index_a2a
                    )

                    # [n_agent, n_step, hidden_dim]
                    if i + 1 < self.num_layers:
                        feat_a_t_dict[i + 1] = torch.cat(
                            (feat_a_t_dict[i + 1], feat_a_now.unsqueeze(1)), dim=1
                        )

            # ! get outputs
            next_token_logits = self.token_predict_head(feat_a_now)
            next_token_logits_list.append(next_token_logits)  # [n_agent, n_token]

            next_token_idx, next_token_traj_all = sample_next_token_traj(
                token_traj=tokenized_agent["token_traj"],
                token_traj_all=tokenized_agent["token_traj_all"],
                sampling_scheme=sampling_scheme,
                # ! for most-likely sampling
                next_token_logits=next_token_logits,
                # ! for nearest-pos sampling
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
                pos_next_gt=tokenized_agent["gt_pos_raw"][:, n_step],  # [n_agent, 2]
                head_next_gt=tokenized_agent["gt_head_raw"][:, n_step],  # [n_agent]
                valid_next_gt=tokenized_agent["gt_valid_raw"][:, n_step],  # [n_agent]
                token_agent_shape=tokenized_agent["token_agent_shape"],  # [n_token, 2]
            )  # next_token_idx: [n_agent], next_token_traj_all: [n_agent, 6, 4, 2]

            diff_xy = next_token_traj_all[:, -1, 0] - next_token_traj_all[:, -1, 3]
            next_token_action_list.append(
                torch.cat(
                    [
                        next_token_traj_all[:, -1].mean(1),  # [n_agent, 2]
                        torch.arctan2(diff_xy[:, [1]], diff_xy[:, [0]]),  # [n_agent, 1]
                    ],
                    dim=-1,
                )  # [n_agent, 3]
            )

            token_traj_global = transform_to_global(
                pos_local=next_token_traj_all.flatten(1, 2),  # [n_agent, 6*4, 2]
                head_local=None,
                pos_now=pos_a[:, t_now],  # [n_agent, 2]
                head_now=head_a[:, t_now],  # [n_agent]
            )[0].view(*next_token_traj_all.shape)

            if not self.training:
                pred_traj_10hz[:, t * 5 : (t + 1) * 5] = token_traj_global[:, 1:].mean(
                    2
                )
                diff_xy = token_traj_global[:, 1:, 0] - token_traj_global[:, 1:, 3]
                pred_head_10hz[:, t * 5 : (t + 1) * 5] = torch.arctan2(
                    diff_xy[:, :, 1], diff_xy[:, :, 0]
                )

            # ! get pos_a_next and head_a_next, spawn unseen agents
            pos_a_next = token_traj_global[:, -1].mean(dim=1)
            diff_xy_next = token_traj_global[:, -1, 0] - token_traj_global[:, -1, 3]
            head_a_next = torch.arctan2(diff_xy_next[:, 1], diff_xy_next[:, 0])
            pred_idx[:, n_step] = next_token_idx

            # ! update tensors for for next step
            pred_valid[:, n_step] = pred_valid[:, t_now]
            # pred_valid[:, n_step] = pred_valid[:, t_now] | mask_spawn
            pos_a = torch.cat([pos_a, pos_a_next.unsqueeze(1)], dim=1)
            head_a = torch.cat([head_a, head_a_next.unsqueeze(1)], dim=1)
            head_vector_a_next = torch.stack(
                [head_a_next.cos(), head_a_next.sin()], dim=-1
            )
            head_vector_a = torch.cat(
                [head_vector_a, head_vector_a_next.unsqueeze(1)], dim=1
            )

            # ! get agent_token_emb_next
            agent_token_emb_next = torch.zeros_like(agent_token_emb[:, 0])
            agent_token_emb_next[veh_mask] = agent_token_emb_veh[
                next_token_idx[veh_mask]
            ]
            agent_token_emb_next[ped_mask] = agent_token_emb_ped[
                next_token_idx[ped_mask]
            ]
            agent_token_emb_next[cyc_mask] = agent_token_emb_cyc[
                next_token_idx[cyc_mask]
            ]
            agent_token_emb = torch.cat(
                [agent_token_emb, agent_token_emb_next.unsqueeze(1)], dim=1
            )

            # ! get feat_a_next
            motion_vector_a = pos_a[:, -1] - pos_a[:, -2]  # [n_agent, 2]
            x_a = torch.stack(
                [
                    torch.norm(motion_vector_a, p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_a[:, -1], nbr_vector=motion_vector_a
                    ),
                ],
                dim=-1,
            )
            # [n_agent, hidden_dim]
            x_a = self.x_a_emb(continuous_inputs=x_a, categorical_embs=categorical_embs)
            # [n_agent, 1, 2*hidden_dim]
            feat_a_next = torch.cat((agent_token_emb_next, x_a), dim=-1).unsqueeze(1)
            feat_a_next = self.fusion_emb(feat_a_next)
            feat_a = torch.cat([feat_a, feat_a_next], dim=1)

        out_dict = {
            # action that goes from [(10->15), ..., (85->90)]
            "next_token_logits": torch.stack(next_token_logits_list, dim=1),
            "next_token_valid": pred_valid[:, 1:-1],  # [n_agent, 16]
            # for step {5, 10, ..., 90} and act [(0->5), (5->10), ..., (85->90)]
            "pred_pos": pos_a,  # [n_agent, 18, 2]
            "pred_head": head_a,  # [n_agent, 18]
            "pred_valid": pred_valid,  # [n_agent, 18]
            "pred_idx": pred_idx,  # [n_agent, 18]
            # for step {5, 10, ..., 90}
            "gt_pos_raw": tokenized_agent["gt_pos_raw"],  # [n_agent, 18, 2]
            "gt_head_raw": tokenized_agent["gt_head_raw"],  # [n_agent, 18]
            "gt_valid_raw": tokenized_agent["gt_valid_raw"],  # [n_agent, 18]
            # or use the tokenized gt
            "gt_pos": tokenized_agent["gt_pos"],  # [n_agent, 18, 2]
            "gt_head": tokenized_agent["gt_heading"],  # [n_agent, 18]
            "gt_valid": tokenized_agent["valid_mask"],  # [n_agent, 18]
            # for shifting proxy targets by lr
            "next_token_action": torch.stack(next_token_action_list, dim=1),
            # "xyh_pred": torch.zeros_like(pos_a.new_zeros(pos_a.shape[0], pos_a.shape[1], 3)),  # 用零张量占位
        }

        if not self.training:  # 10hz predictions for wosac evaluation and submission
            out_dict["pred_traj_10hz"] = pred_traj_10hz
            out_dict["pred_head_10hz"] = pred_head_10hz
            pred_z = tokenized_agent["gt_z_raw"].unsqueeze(1)  # [n_agent, 1]
            out_dict["pred_z_10hz"] = pred_z.expand(-1, pred_traj_10hz.shape[1])

        return out_dict
