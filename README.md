# SplatKing-Converter
将 SplatKing 扫描得到的 LiDAR / COLMAP 文本模型数据，转换为 Postshot 等高斯计算/训练工具可用的数据格式。

## 使用方法

Windows PowerShell 示例（把路径替换成你自己的即可）：

```powershell
py c:\Users\Admin\Documents\IPHONE\colmap_coord_convert.py `
  --in  "...\COLMAP_Text_Model\sparse\0" `
  --out "...\COLMAP_Text_Model\sparse\0_lidar" `
  --camera-preset none `
  --world-preset none `
  --camera-euler-zyx-deg 180 180 0 `
  --ply "...\lidar_pointcloud_world_xyz.ply" `
  --min-views-per-point 1 `
  --max-points 20000
```

## 参数说明

- --in：输入 COLMAP 文本模型目录（包含 cameras.txt / images.txt / points3D.txt）
- --out：输出目录（会生成转换后的 COLMAP 文本模型/点云相关文件）
- --camera-preset：相机预设（这里用 none）
- --world-preset：世界坐标预设（这里用 none）
- --camera-euler-zyx-deg：相机欧拉角（ZYX 顺序，单位：度），示例为 180 180 0
- --ply：SplatKing 导出的 LiDAR 点云（world xyz）PLY 文件路径
- --min-views-per-point：过滤点的最小观测次数
- --max-points：最多保留的点数（用于下采样/裁剪）
