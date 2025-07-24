import numpy as np
import os
import pickle
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Manager
from queue import Empty
from waymo_open_dataset.protos import sim_agents_submission_pb2
from google.protobuf import text_format
from waymo_open_dataset.protos import sim_agents_metrics_pb2
import fast_sim_agents_metrics.metrics as sim_agents_metric_api
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import fast_sim_agents_metrics.scenario_converter as scenario_converter

def worker(rank, scenario_dir, predict_dir, file_queue, sim_agent_eval_config, result_queue):
    # 设置当前进程使用的GPU
    torch.cuda.set_device(rank)
    
    while True:
        try:
            # 从队列中获取下一个文件，如果队列为空则退出
            file = file_queue.get_nowait()
            try:
                with open(os.path.join(scenario_dir, file), 'rb') as f:
                    scenario = pickle.load(f)
                with open(os.path.join(predict_dir, file), 'rb') as f:
                    predict = pickle.load(f)    
                predict['agent_id'] = torch.tensor(predict['agent_id'],device=rank)
                predict['simulated_states'] = torch.tensor(predict['simulated_states'],device=rank)
                gt_scenario = scenario_converter.extract_gt_scenario(scenario, device=rank)
                scenario_metrics = sim_agents_metric_api.compute_scenario_metrics_for_bundle(sim_agent_eval_config, gt_scenario, predict)
                result_queue.put(scenario_metrics)
            except Exception as e:
                print(f"Error processing file {file} on GPU {rank}: {str(e)}")
        except Empty:
            break

def main():
    scenario_dir = '/data0/datasets/waymo_open_dataset_motion_v_1_3_0/separated_senario/validation'
    predict_dir = '/data1/lqf/MMSim++/MMSim/validation_results/origin_20_old'
    files = os.listdir(predict_dir)
    
    #assert len(os.listdir(scenario_dir)) == len(os.listdir(predict_dir))
    with open('/data1/lqf/conda_envs/smart/lib/python3.9/site-packages/waymo_open_dataset/wdl_limited/sim_agents_metrics/challenge_2024_config.textproto','r') as f:
        sim_agent_eval_config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
        text_format.Parse(f.read(), sim_agent_eval_config)
    
    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    
    # 创建进程管理器
    manager = Manager()
    
    # 创建文件队列和结果队列
    file_queue = manager.Queue()
    result_queue = manager.Queue()
    
    # 将所有文件放入队列
    for file in files:
        file_queue.put(file)
    
    # 启动多个进程
    processes = []
    for i in range(num_gpus):
        p = Process(target=worker, args=(i, scenario_dir, predict_dir, file_queue, sim_agent_eval_config, result_queue))
        p.start()
        processes.append(p)
    
    # 收集结果
    results = []
    with tqdm(total=len(files), desc="Processing files") as pbar:
        while len(results) < len(files):
            result = result_queue.get()
            results.append(result)
            pbar.update(1)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 计算最终结果
    final_result = {}
    for metric_name in ['metametric',
                        'linear_speed_likelihood','linear_acceleration_likelihood','angular_speed_likelihood',
                        'angular_acceleration_likelihood','distance_to_nearest_object_likelihood','distance_to_road_edge_likelihood',
                        'collision_indication_likelihood','time_to_collision_likelihood','simulated_collision_rate',
                        'simulated_offroad_rate','average_displacement_error','min_average_displacement_error']:
        final_result[metric_name] = 0
        for result in results:
            final_result[metric_name] += result[metric_name]
        final_result[metric_name] /= len(results)

    print(final_result)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()