# PathCRF (third-party code)

Source: https://github.com/hyunsungkim-ds/pathcrf, commit 1b7c6f8
Paper: Kim H., Lee K., Seo S., Ko S.-K., Yoon J., Park C. "PathCRF: Ball-Free Soccer
Event Detection via Possession Path Inference from Player Trajectories". KDD 2026.
arXiv:2602.12080. DOI: 10.1145/3770855.3818152
License: Mozilla Public License 2.0 (see LICENSE in this folder)

What is included: the code needed for inference (inference.py, models/, datatools/,
set_transformer/) and the pretrained model saved/140 (Dynamic Masked CRF:
args.json + state_dict_best_acc.pt), trained by the authors on the public
DFL/IDSSE dataset (Bassek et al., 2025, CC BY 4.0).
Modifications: none. The files are identical to the ones at the commit above.
