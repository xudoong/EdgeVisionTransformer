from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class ModelInfo:
    b_macs : float = 0.0
    acc : float =  0.0
    m_params : float = 0.0


modelinfo_dict = dict(
    deit_base = ModelInfo(17.7, 81.8),
    deit_small = ModelInfo(4.64, 79.9),
    deit_tiny = ModelInfo(1.28, 72.2),
    t2t_vit_14 = ModelInfo(4.8, 81.5),
    t2t_vit_12 = ModelInfo(1.8, 76.5),
    t2t_vit_10 = ModelInfo(1.5, 75.2),
    t2t_vit_7 = ModelInfo(1.1, 71.7),
    swin_base = ModelInfo(15.4, 83.5),
    swin_small = ModelInfo(8.7, 83),
    swin_tiny = ModelInfo(4.5, 81.3),
    autoformer_base = ModelInfo(11, 82.4),
    autoformer_small = ModelInfo(5.1, 81.7),
    autoformer_tiny = ModelInfo(1.3, 74.7),
    efficientnet_b7 = ModelInfo(37, 84.3),
    efficientnet_b6 = ModelInfo(19, 84),
    efficientnet_b5 = ModelInfo(9.9, 83.6),
    efficientnet_b4 = ModelInfo(4.2, 82.9),
    efficientnet_b3 = ModelInfo(1.8, 81.6),
    efficientnet_b2 = ModelInfo(1.0, 80.1),
    efficientnet_b1 = ModelInfo(0.7, 79.1),
    efficientnet_b0 = ModelInfo(0.39, 77.1),
    resnet_152 = ModelInfo(11, 77.8),
    resnet_101 = ModelInfo(7.9, 77.4),
    resnet_50 = ModelInfo(4.1, 76),
    mobilenet_v2 = ModelInfo(0.3, 72),
    mobilenet_v3_large = ModelInfo(0.22, 75.6),
    proxyless_mobile = ModelInfo(0.32, 74.6)
)


deit_list = ['deit_tiny', 'deit_small', 'deit_base']
t2t_vit_list = ['t2t_vit_7', 't2t_vit_10', 't2t_vit_12', 't2t_vit_14']
swin_list = ['swin_tiny', 'swin_small', 'swin_base']
autoformer_list = ['autoformer_base', 'autoformer_small', 'autoformer_tiny']
efficientnet_list = [f'efficientnet_b{v}' for v in range(0, 8)]
resnet_list = ['resnet_50', 'resnet_101', 'resnet_152']
mobilenet_list = ['mobilenet_v2', 'mobilenet_v3_large']
proxyless_mobile_list = ['proxyless_mobile']

plt.plot([modelinfo_dict[x].b_macs for x in deit_list], 
         [modelinfo_dict[x].acc for x in deit_list],
         label='deit', c='#0099ff', marker='^')
plt.plot([modelinfo_dict[x].b_macs for x in t2t_vit_list], 
         [modelinfo_dict[x].acc for x in t2t_vit_list],
         label='t2t_vit', c='#4d4dff', marker='^')
plt.plot([modelinfo_dict[x].b_macs for x in swin_list], 
         [modelinfo_dict[x].acc for x in swin_list], 
         label='swin transformer', c='#944dff', marker='^')
plt.plot([modelinfo_dict[x].b_macs for x in autoformer_list], 
         [modelinfo_dict[x].acc for x in autoformer_list], 
         label='autoformer', c='#0099cc', marker='^')
    
plt.plot([modelinfo_dict[x].b_macs for x in efficientnet_list], 
         [modelinfo_dict[x].acc for x in efficientnet_list], 
         label='efficientnet', c='#cc3300', marker='o')
plt.plot([modelinfo_dict[x].b_macs for x in resnet_list], 
         [modelinfo_dict[x].acc for x in resnet_list], 
         label='resnet', c='#e67300', marker='o')
plt.plot([modelinfo_dict[x].b_macs for x in mobilenet_list], 
         [modelinfo_dict[x].acc for x in mobilenet_list], 
         label='mobilenet', c='#ffaa00', marker='o')
plt.plot([modelinfo_dict[x].b_macs for x in proxyless_mobile_list], 
         [modelinfo_dict[x].acc for x in proxyless_mobile_list], 
         label='proxyless_mobile', c='#ff4d4d', marker='o')

plt.title('Model MACs and Accuracy')
plt.xlabel('Billion MACs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig('tmp.png')