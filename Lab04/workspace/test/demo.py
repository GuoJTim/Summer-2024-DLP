import os
import argparse
import pandas as pd
import pandas.api.types
from math import log10
from sklearn.metrics import mean_squared_error

def PSNR(mse, data_range=255.):
    psnr = 20 * log10(data_range) - 10 * log10(mse)
    return psnr

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    del solution[row_id_column_name]
    del solution['Usage']
    del submission[row_id_column_name]

    
    gt = solution.to_numpy()
    
    submission = submission.to_numpy()
    
    if submission.shape != gt.shape:
        raise KeyError(f'{gt.shape} Got the submission with shape {submission.shape}, which did not match the shape {gt.shape}')
    
    mse_list = [mean_squared_error(gt[i], submission[i]) for i in range(gt.shape[0])]
    
    psnr_list = []
    for i, mse in enumerate(mse_list):
        if i % 630 != 0 and i % 630 < 601:
            psnr_list.append(PSNR(mse))
    
    public_score = sum(psnr_list[:1800])/len(psnr_list[:1800])
    private_score = sum(psnr_list[1800:])/len(psnr_list[1800:])
    

    return public_score, private_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--gt_path', type=str, default='./gt.csv')
    parser.add_argument('--submission_path', type=str, default='./demo/submission.csv')
    args = parser.parse_args()
    solution = pd.read_csv(args.gt_path)
    submission = pd.read_csv(args.submission_path)
    public_score, private_score = score(solution=solution, submission=submission, row_id_column_name='id')
    print('Public score: ', public_score)
    print('Private score: ', private_score)
    