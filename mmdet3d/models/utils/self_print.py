import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import cv2
import os
import torch
import numpy as np
import shutil

from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import numpy as np
from pathlib import Path
import tqdm


def print2file(content, suffix="", end=".json", mode="w"):
    f = open("/mnt/data/exps_logs/out_" + suffix + end, mode=mode)
    print(content, file=f)


# def feats_to_img(feats, base_path, suffix="out", **kwargs):

#     base_path = os.path.join(base_path, suffix)

#     base_path = os.path.join(base_path, suffix)
#     if os.path.exists(base_path):
#         pass
#     else:
#         # os.mkdir(base_path)
#         os.makedirs(base_path)

#     bs, c, h, w = feats.shape
#     assert bs >= 1
#     # feats = feats[0].detach().cpu().numpy()
#     feats = feats[0]
#     if "boxes" in kwargs.keys():
#         gt_boxes = kwargs["boxes"]


#     for idx, feat in enumerate(feats):

#         heatmapshow = None
#         # heatmapshow = cv2.normalize(
#         #     feats,
#         #     heatmapshow,
#         #     alpha=0,
#         #     beta=255,
#         #     norm_type=cv2.NORM_MINMAX,
#         #     dtype=cv2.CV_8U,
#         # )

#         heatmapshow = (
#             feats.mul(255)
#             .add_(0.5)
#             .clamp_(0, 255)
#             .permute(1, 2, 0)
#             .to("cpu", torch.uint8)
#             .numpy()
#         )
#         heatmapshow = heatmapshow.astype(np.uint8)
#         heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)

#         for box in gt_boxes:
#             cv2.circle(heatmapshow, box[:2] / 8, 3, "black", 0)


#         cv2.imwrite(
#             f"{base_path}/gray_scale_tensor{idx}.png",
#             heatmapshow,
#         )
def mkdir_by_path(base_path):
    if os.path.exists(base_path):
        pass
    else:
        # os.mkdir(base_path)
        os.makedirs(base_path)


def feats_to_img(feats, base_path, suffix="out", indice=-1, sample_idx=None, **kwargs):
    if feats is None:
        return
    feats = feats[0] if isinstance(feats, list) else feats

    base_path = os.path.join(base_path, suffix)
    mkdir_by_path(base_path=base_path)

    bs, c, h, w = feats.shape
    assert bs >= 1
    feats = feats[0].detach().cpu().numpy()

    if indice == 0:
        # print()
        if sample_idx is not None:
            feature = feats[0, ...]
            feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-5)
            feature = np.uint8(255 * feature)  # 转换为 8-bit 灰度图格式
            cv2.imwrite(f"{base_path}/{sample_idx}.png", feature)
            print(f"process : {base_path}/{sample_idx}.png")
            # plt.imsave(
            #     f"{base_path}/{sample_idx}.png",
            #     feats[0, ...],
            #     cmap="gray",
            # )
        else:
            plt.imsave(
                f"{base_path}/gray_scale_tensor{indice}.png",
                feats[0, ...],
                cmap="gray",
            )
    else:
        save_multi_channel_path = os.path.join(base_path, sample_idx)
        mkdir_by_path(save_multi_channel_path)
        for idx, feat in enumerate(feats):

            # # print()
            # plt.imsave(
            #     f"{save_multi_channel_path}/gray_scale_tensor{idx}.png",
            #     feat,
            #     cmap="gray",
            # )

            feature = (feat - feat.min()) / (feat.max() - feat.min() + 1e-5)
            feature = np.uint8(255 * feature)  # 转换为 8-bit 灰度图格式
            cv2.imwrite(
                f"{save_multi_channel_path}/gray_scale_tensor{idx}.png", feature
            )


def feats_to_img_boxes(feats, base_path, suffix="out", **kwargs):
    feats = feats[0] if isinstance(feats, list) else feats
    import os

    boxes = kwargs["boxes"]
    points = [(box[0] / 8, box[1] / 8) for box in boxes]

    base_path = os.path.join(base_path, suffix)
    if os.path.exists(base_path):
        pass
    else:
        # os.mkdir(base_path)
        os.makedirs(base_path)

    bs, c, h, w = feats.shape
    assert bs >= 1
    feats = feats[0].detach().cpu().numpy()

    for idx, feat in enumerate(feats):

        for point in points:
            plt.plot(point[0], point[1], "bo")  # 'bo'表示蓝色圆点

        # print()
        plt.imsave(
            f"{base_path}/gray_scale_tensor{idx}.png",
            feat,
            cmap="gray",
        )


def save_feature_to_img_cam(
    features, idx, visual_path=None, suffix="heatmap", sample_idx=None
):
    features = features[0] if isinstance(features, list) else features
    assert sample_idx is not None
    # visual_path = visual_path.split("LIDAR_TOP/")[1].split(".pcd.bin")[0]
    # features=self.get_single_feature() # 返回一个指定层输出的特征图,属于四维张量[batch,channel,width,height]
    if idx == 0:
        feature = features[
            0, idx, :, :
        ]  # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
        feature = feature.view(
            feature.shape[0], feature.shape[1]
        )  # batch为1，所以可以直接view成二维张量

        feature = feature.cpu().data.numpy()  # 转为numpy

        # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
        # feature = cv2.resize(feature, (256, 256))
        feature = (feature - np.amin(feature)) / (
            np.amax(feature) - np.amin(feature) + 1e-5
        )  # 注意要防止分母为0！
        feature = np.uint8(
            255 * feature
        )  # np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行
        feature = cv2.resize(feature, (256, 256))
        heatmap = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
        mkdir_by_path(
            os.path.join(visual_path, suffix)
        )  # 创建保存文件夹，以选定可视化层的序号命名
        print(visual_path)
        superimposed_img = heatmap  # + img
        cv2.imwrite(
            visual_path + f"/{suffix}/" + sample_idx + ".jpg", superimposed_img
        )  # 保存当前层输出的每个channel上的特征图为一张图像
        # cv2.imwrite(path  + str(i) + '.jpg', superimposed_img)
    else:
        for i in range(features.shape[1]):
            print("----------", features.shape)
            feature = features[
                :, i, :, :
            ]  # 在channel维度上，每个channel代表了一个卷积核的输出特征图，所以对每个channel的图像分别进行处理和保存
            feature = feature.view(
                feature.shape[1], feature.shape[2]
            )  # batch为1，所以可以直接view成二维张量

            feature = feature.cpu().data.numpy()  # 转为numpy

            # 根据图像的像素值中最大最小值，将特征图的像素值归一化到了[0,1];
            # img = cv2.resize(img, (256, 256))
            print("type: ", type(feature))
            print("shape: ", feature.shape)
            # feature = cv2.resize(feature, (256, 256))
            feature = (feature - np.amin(feature)) / (
                np.amax(feature) - np.amin(feature) + 1e-5
            )  # 注意要防止分母为0！
            feature = np.uint8(
                255 * feature
            )  # np.round(feature * 255) # [0, 1]——[0, 255],为cv2.imwrite()函数而进行
            feature = cv2.resize(feature, (256, 256))
            heatmap = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
            path = os.path.join("outfeature/Cam/", visual_path)  #   visaul_path+idx
            mkdir_by_path(base_path=path)  # 创建保存文件夹，以选定可视化层的序号命名
            print(path)
            superimposed_img = heatmap * 0.4  # + img
            cv2.imwrite(
                path + "/" + str(i) + ".jpg", superimposed_img
            )  # 保存当前层输出的每个channel上的特征图为一张图像
            # cv2.imwrite(path  + str(i) + '.jpg', superimposed_img)


def to_heatmap(base_path):
    gray_images_path = os.path.join(base_path, "img_feats_none")
    heatmap_path = os.path.join(base_path, "heatmap")
    mkdir_by_path(heatmap_path)

    cmap = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])

    # 获取文件夹内所有图像文件
    image_files = [
        f
        for f in os.listdir(gray_images_path)
        if f.endswith(".png") or f.endswith(".jpg")
    ]

    # 遍历每个图像文件
    with tqdm.tqdm(image_files) as processor:
        for image_file in processor:
            # 读取灰度图
            img = Image.open(os.path.join(gray_images_path, image_file)).convert("L")
            image_array = np.array(img)

            # 将灰度图从 [0, 255] 映射到 [0, 1] 进行处理
            image_array = image_array.astype(np.float32)
            _image_array = image_array
            # image_array = (image_array - np.amin(image_array)) / (np.amax(image_array) - np.amin(image_array) + 1e-5)

            image_array = np.uint8(255 * image_array)
            # COLORMAP_JET or COLORMAP_RAINBOW or COLORMAP_TURBO or COLORMAP_WINTER
            heatmap = cv2.applyColorMap(image_array, cv2.COLORMAP_RAINBOW)

            heatmap2 = cv2.applyColorMap(image_array, cv2.COLORMAP_JET)
            heatmap = heatmap * 0.8 + heatmap2 * 0.2

            output_path = os.path.join(heatmap_path, "heatmap_" + image_file)
            cv2.imwrite(output_path, heatmap)

            processor.set_description(output_path)

        # print(f"Saved heatmap for {image_file} at {output_path}")

        # 遍历每个图像文件
        # for image_file in processor:
        #     fig = plt.figure()
        #     # 读取灰度图
        #     img = Image.open(os.path.join(gray_images_path, image_file)).convert("L")
        #     image_array = np.array(img)
        #     plt.imshow(image_array, cmap="coolwarm")
        #     plt.colorbar()  # 添加颜色条

        #     output_path = os.path.join(heatmap_path, "heatmap_" + image_file)
        #     fig.savefig(output_path)
        #     # plt.show()

        #     plt.close()


def to_heatmap_cv2(base_path):
    gray_images_path = os.path.join(base_path, "img_feats_none")
    heatmap_path = os.path.join(base_path, "heatmap_virdis")
    mkdir_by_path(heatmap_path)

    image_files = [
        f
        for f in os.listdir(gray_images_path)
        if f.endswith(".png") or f.endswith(".jpg")
    ]

    # 遍历每个灰度图文件
    for image_file in tqdm.tqdm(image_files):

        # 读取灰度图像
        img = cv2.imread(
            os.path.join(gray_images_path, image_file), cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            print(f"Unable to read {image_file}. Skipping...")
            continue

        img = img.astype(np.float32)
        feature = img / 255.0
        # 确保数值在 [0, 1] 范围内，不超过 1 或小于 0、
        # feature = (feature - feature.min()) / (feature.max() - feature.min())
        feature = np.clip(feature, 0, 1)
        feature = np.uint8(255 * feature)

        if feature.max() == feature.min():  # 如果值没有变化（都是相同的），跳过处理
            print(f"Skipping {image_file} due to zero variance.")
            continue

        # COLORMAP_JET or COLORMAP_RAINBOW or COLORMAP_TURBO or COLORMAP_WINTER
        heatmap = cv2.applyColorMap(feature, cv2.COLORMAP_VIRIDIS)

        # 可选：如果要将热力图与原图叠加，可以调整比例
        superimposed_img = heatmap

        # 保存当前生成的热力图
        cv2.imwrite(
            os.path.join(heatmap_path, "heatmap_" + image_file), superimposed_img
        )


def find_files_with_sample_idx(base_path, sample_idx):
    # Ensure the base_path directory exists
    if not os.path.exists(base_path):
        print(f"Error: Base path {base_path} does not exist.")
        return []

    # Get all files in the base_path directory
    all_files = os.listdir(base_path)

    # Filter files that contain the substring sample_idx
    matching_files = [file for file in all_files if sample_idx in file]

    # Construct full paths for the matching files
    matching_file_paths = [os.path.join(base_path, file) for file in matching_files]
    if matching_files.__len__() == 0:
        return

    return matching_file_paths[0]


def cp_sample_idx_to_path(base_path, sample_idx, to_path, feat_suffix="img_feats_none"):
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    base_path_file_name = Path(base_path).name
    to_path = os.path.join(to_path, "feats_vis", sample_idx, base_path_file_name)

    # if os.path.exists(to_path):
    #     shutil.rmtree(to_path)

    mkdir_by_path(to_path)

    src_file_base = os.path.join(base_path, "vis", feat_suffix)

    src_file = find_files_with_sample_idx(src_file_base, sample_idx=sample_idx)

    if src_file is not None:
        dst_file = os.path.join(
            to_path, base_path_file_name + "_" + os.path.basename(src_file)
        )

        # Copy the file from source to destination
        shutil.copy(src_file, dst_file)


def cp_det_to_path(base_path, sample_idx, to_path):
    if not os.path.exists(to_path):
        os.makedirs(to_path)

    base_path_file_name = Path(base_path).name
    to_path = os.path.join(to_path, "det_vis", sample_idx, base_path_file_name)

    # if os.path.exists(to_path):
    #     shutil.rmtree(to_path)
    #     # shutil.rmtree(os.path.join(to_path, "det_vis"))

    mkdir_by_path(to_path)

    src_file_base = os.path.join(base_path, "det")

    src_file = find_files_with_sample_idx(src_file_base, sample_idx=sample_idx)

    if src_file is not None:
        dst_file = os.path.join(
            to_path, base_path_file_name + "_" + os.path.basename(src_file)
        )

        # Copy the file from source to destination
        shutil.copy(src_file, dst_file)


def main():
    base_path_lists = [
        "/mnt/data/exps/D3PD/hybirdbev-review/bevdet",
        "/mnt/data/exps/D3PD/hybirdbev-review/bevdet-4d-512-1408",
        "/mnt/data/exps/D3PD/hybirdbev-review/bevdet-geomim",
        "/mnt/data/exps/D3PD/hybirdbev-review/bevdet-long-depth",
        "/mnt/data/exps/D3PD/hybirdbev-review/bevdet-long-stereo",
        "/mnt/data/exps/D3PD/hybirdbev-review/bevdet4d",
        "/mnt/data/exps/D3PD/hybirdbev-review/bevdet4d-depth",
        # "/mnt/data/exps/D3PD/hybirdbev-review/outrs-stereo-L",
    ]

    sample_idx_lists = [
        # "6a467240b14745cd8ab11701bd4f08d3",
        # "0dce353f4164414e996aee2520f185af",
        # "0e8e2f64b807428ca4ed5bc33c996389",
        # "0e8782aa721545caabc7073d32fb1fb1",
        # new add
        # "bb05f4d61f124d5f9c520967ce0544f6",
        # "4176983e28014670b8d4599c5e70166a",
        # "3219230d0e14480c9089c7c49e339a83",
        # "557954f121394533b582e88eac8017ad",
        # simple test compare
        # "3c18f85f037744b6ae4c8a6f8dc578c2",
        # "00a9b6fdf99143c4aabaaca13004c2be",
        # "0a0d6b8c2e884134a3b48df43d54c36a",
        # "0a8dee95c4ac4ac59a43af56da6e589f",
        "0aa136a1b6f54e8faed7e1d08ebfaa82",
        "0adea7e136624021ba81b30a63976bd4",
        "0aa136a1b6f54e8faed7e1d08ebfaa82",
    ]

    det_base_path = []

    if os.path.exists("/mnt/data/det_vis"):
        shutil.rmtree("/mnt/data/det_vis")
    if os.path.exists("/mnt/data/feats_vis"):
        shutil.rmtree("/mnt/data/feats_vis")

    for base in tqdm.tqdm(base_path_lists):
        base_vis_path = f"{base}/vis"

        # to_heatmap(base_vis_path)

        # to_heatmap_cv2(base_vis_path)

        for s_idx in sample_idx_lists:
            cp_sample_idx_to_path(
                base,
                sample_idx=s_idx,
                to_path="/mnt/data",
                feat_suffix="heatmap",
            )

            cp_det_to_path(base_path=base, sample_idx=s_idx, to_path="/mnt/data")


if __name__ == "__main__":
    main()
