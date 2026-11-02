深度学习框架目标检测算法如何使用Yolov8汽车轮胎轮毂缺陷数据集 检测识别汽车轮胎钢圈缺陷数据集 建立深度学习汽车汽车轮胎轮毂缺陷识别
@[toc]
以下文字及代码仅供参考。
汽车轮胎钢圈轮毂缺陷检测数据集

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/86372ba60aa44676b125783617b77b75.png)


轮毂缺陷检测YOLO数据集

---![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2049bd6d6eb648f4ab74fb1df8abf4b8.png)


### 🛠️ 轮毂缺陷检测 数据集统计表（共7 类）

| 类别名称                   | 图片数量 | 标注框数量 | 训练集图片数（约80%） | 验证集图片数（约20%） |
|----------------------------|----------|------------|------------------------|------------------------|
| Peeling（剥落）            | 460      | 1276       | 368                    | 92                     |
| Out_Of_Roundness（失圆）   | 37       | 48         | 30                     | 7                      |
| Brake_Thermal_Cracks（制动热裂纹） | 70       | 391        | 56                     | 14                     |
| Contact_Fatigue_Cracks（接触疲劳裂纹） | 300     | 817        | 240                    | 60                     |
| Blue hole（蓝孔）          | 311      | 483        | 249                    | 62                     |
| Rollerpile（辊堆缺陷）     | 301      | 320        | 241                    | 60                     |
| flat（扁平化）             | 293      | 315        | 234                    | 59                     |
| **总计**                   | **1445** | **3650**   | **1156 (80%)**         | **289 (20%)**          |

---

###  数据说明：

- 数据集总图片数：**1445 张**
- 总标注框数：**3650 个**
- 按照 **8:2 的比例划分为训练集和验证集**
- 每张图片中可能包含多个缺陷目标（如多个标注框），因此标注框总数大于图片总数
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0d71fece091746069747f7af30a186a4.png)

---

：
- 生成 `.yaml` 配置文件用于 YOLO 模型训练
- 自动生成 train/val 文件列表（txt/json）
轮毂缺陷检测的数据集结构， **7类轮毂缺陷目标检测任务** 

- 数据组织方式（YOLO格式）
- `data.yaml` 配置文件
- 模型训练代码
- 推理预测代码
- 性能评估方法

---

## ✅ 一、数据准备与组织结构

确保你的数据集按照以下目录结构进行组织：

```
hub_defect_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### 示例说明：

- `images/train/`：存放训练图像（1156张）
- `labels/train/`：每个图像对应的 `.txt` 标注文件
- `images/val/`：验证图像（289张）
- `labels/val/`：对应标注文件

> 注：每张图像必须有一个同名的 `.txt` 文件，内容格式为：`class_id center_x center_y width height`，且所有坐标归一化到 [0,1]。

---

## ✅ 二、创建 `data.yaml` 文件

在项目根目录下创建 `data.yaml` 文件，内容如下：

```yaml
train: ./hub_defect_dataset/images/train
val: ./hub_defect_dataset/images/val

nc: 7
names: ['Peeling', 'Out_Of_Roundness', 'Brake_Thermal_Cracks', 'Contact_Fatigue_Cracks', 'Blue hole', 'Rollerpile', 'flat']
```

---

## ✅ 三、安装依赖库

```bash
pip install ultralytics opencv-python matplotlib numpy
```

---

## ✅ 四、模型训练代码（Python脚本）

```python
from ultralytics import YOLO

# 加载预训练模型或从配置文件初始化
model = YOLO('yolov8s.yaml')  # 可换成 yolov8n / m / l / x

# 开始训练
results = model.train(
    data='./hub_defect_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='hub_defect_train',
    project='runs/hub_defect',
    exist_ok=True,
    device=0 if torch.cuda.is_available() else 'cpu'
)

# 保存最佳权重
best_weights_path = model.save('./hub_defect_best.pt')
print(f"Best weights saved to {best_weights_path}")
```

---

## ✅ 五、模型验证与性能评估

```python
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('./hub_defect_best.pt')

# 在验证集上评估
metrics = model.val()

# 打印关键指标
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"Recall: {metrics.box.recall.mean():.4f}")
print(f"Precision: {metrics.box.precision.mean():.4f}")
```

---

## ✅ 六、单图推理 + 结果可视化

```python
from ultralytics import YOLO
import cv2

def detect_and_show(image_path):
    model = YOLO('./hub_defect_best.pt')

    results = model.predict(source=image_path, conf=0.25)
    
    for result in results:
        im_array = result.plot()
        im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
        cv2.imshow("Detection Result", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 测试一张图片
detect_and_show('path/to/test_image.jpg')
```

---

## ✅ 七、批量推理并保存结果

```python
from ultralytics import YOLO

model = YOLO('./hub_defect_best.pt')

# 批量处理指定文件夹下的图片，并保存结果
model.predict(
    source='./hub_defect_dataset/images/val',
    save=True,
    project='runs/hub_defect/predictions',
    name='val_results',
    conf=0.25
)
```

---

## ✅ 八、生成分类报告与混淆矩阵（进阶）

需要将真实标签和预测标签提取出来后使用 `sklearn.metrics` 工具：

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 假设 y_true 和 y_pred 是两个列表，包含真实类别和预测类别 ID
class_names = ['Peeling', 'Out_Of_Roundness', 'Brake_Thermal_Cracks', 'Contact_Fatigue_Cracks', 'Blue hole', 'Rollerpile', 'flat']

# 分类报告
print(classification_report(y_true, y_pred, target_names=class_names))

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
```

---

## ✅ 九、建议优化方向

由于某些类别样本较少（如 Out_Of_Roundness 仅37张），建议：
- 使用 **数据增强**（mosaic、hsv等）
- 设置 **类别权重 loss_weight**
- 尝试 **过采样 / 重采样策略**
- 使用 **更强大的模型架构**（如 yolov8m / yolov8l）

---
以上文字仅供参考和学习，



