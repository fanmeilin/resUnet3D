# resUnet3D

## 几点说明：

### 1，结构信息

完全按照对称结构还原resnet的残差基础上每一层有相加
后续对于pool的是否保留
对output_padding的在最后的部分是否会有丢失情况有待考量
另外可以再改进的地方

### 2，其他模块

#### 改进点

- 可以考虑注意力，transfomer，传统去噪方法滤波等改进**去噪**；
- 预训练，蒸馏来改进**分类**；
- 对图像增加泊松噪声的挖点；

#### 商榷点

- 对比pooling，bn层是否需要保留
- 对比残差块中tranconv和conv 可验证tranconv的效果是否会比conv效果好
- 当尺寸很小时，是否删除512层次

## 训练

### 预训练模型

```misc
--model resnet --model_depth 18 --n_pretrain_classes 700
```

#### 官方代码说明

Fine-tune fc layers of a pretrained model (~/data/models/resnet-50-kinetics.pth) on UCF-101.

```bash
python main.py --root_path ~/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \\
--result_path results --dataset ucf101 --n_classes 101 --n_pretrain_classes 700 \\
--pretrain_path models/resnet-50-kinetics.pth --ft_begin_module fc \\
--model resnet --model_depth 50 --batch_size 128 --n_threads 4 --checkpoint 5
```

参数：run-》edit configuration

```python
--root_path ./datasets/ADNI_denoise --noise_path gaussian15 --clean_path FD --result_path results --pretrain_path ./weights/r3d18_KM_200ep.pth --model resUnet3D --model_depth 18 --batch_size 128  --n_epochs 10 --checkpoint 10 
```

查看日志

```
tensorboard --logdir=./datasets/ADNI_denoise/results
```



### 下一步工作

1. 预训练模型的导入
2. 尺寸的确定
3. 添加噪声
4. 编写dataset，loss等函数
5. 训练
6. 预训练模型 视频3通道 腾讯医学待考虑

视频的预训练模型的载入是在去噪网络还是分类网络

预训练时，视频预训练网络应该用在去噪网络上，再把encoder部分载入resnet微调分类

蒸馏时，视频预训练模型应该用在分类网络上叭

还有很多冻结参数，网络结构可以调整

#### results_gaussian15_layer10_lr0.1

![image-20220226163608249](C:\Users\mulin\AppData\Roaming\Typora\typora-user-images\image-20220226163608249.png)

#### results_gaussian15_layer10_lr0.6

![image-20220226163524607](C:\Users\mulin\AppData\Roaming\Typora\typora-user-images\image-20220226163524607.png)

#### results_gaussian15_layer10_lr0.01过拟合

![image-20220226163502624](C:\Users\mulin\AppData\Roaming\Typora\typora-user-images\image-20220226163502624.png)

#### results_gaussian15_layer18_lr0.1

![image-20220226163334071](C:\Users\mulin\AppData\Roaming\Typora\typora-user-images\image-20220226163334071.png)

#### results_gaussian15_layer18_lr0.1drop512

![image-20220226163254810](C:\Users\mulin\AppData\Roaming\Typora\typora-user-images\image-20220226163254810.png)

#### results_gaussian15_layer18_lr0.1_drop512_optimAdam

![image-20220226163208602](C:\Users\mulin\AppData\Roaming\Typora\typora-user-images\image-20220226163208602.png)

#### results_gaussian15_layer34_lr0.1drop512

![image-20220226163043318](C:\Users\mulin\AppData\Roaming\Typora\typora-user-images\image-20220226163043318.png)