"""
https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf
"""
from captum.attr import GradientShap # SHAP + IG with a gaussian distributed baseline
from captum.attr import DeepLiftShap # SHAP + DeepLift
from captum.attr import GradientAttribution # 