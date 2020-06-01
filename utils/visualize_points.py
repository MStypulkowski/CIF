import argparse
import json
import os
import typing as t

import cv2
import numpy as np
import requests
import torch
import tqdm


def decode_image(byte_data: t.List[float]) -> np.ndarray:
    byte_data = np.asarray(byte_data, dtype=np.uint8)[..., np.newaxis]
    img = cv2.imdecode(byte_data, cv2.IMREAD_COLOR)
    return img


def standardize_bbox(
    pcl: np.ndarray, points_per_object: int, return_point_indices: bool = False
) -> t.Union[np.ndarray, t.Tuple[np.ndarray, np.ndarray]]:
    point_indices = np.arange(len(pcl))

    pcl = pcl[:points_per_object]  # n by 3
    point_indices = point_indices[:points_per_object]

    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.0
    scale = np.amax(maxs - mins)
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    if return_point_indices:
        return result, point_indices
    return result


xml_head = """
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="1000"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
            </transform>
            <float name="fov" value="25"/>

            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="ldrfilm">
                <integer name="width" value="640"/>
                <integer name="height" value="480"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>

        <bsdf type="roughplastic" id="surfaceMaterial">
            <float name="alpha" value="0.1"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="0.63,0.61,0.58"/> <!-- default 0.5 -->
        </bsdf>

        <shape type="rectangle">
            <transform name="toWorld">
                <lookat origin="0,-11,5" target="0,0,0" /> 
                <scale x="0.5" y="0.5" z="0.5" />
            </transform>
            <emitter type="area">
                <spectrum name="radiance" value="40"/>
            </emitter> 
        </shape>

    """

xml_wide_head = """
    <scene version="0.6.0">
        <integrator type="path">
            <integer name="maxDepth" value="-1"/>
        </integrator>
        <sensor type="perspective">
            <float name="farClip" value="10000"/>
            <float name="nearClip" value="0.1"/>
            <transform name="toWorld">
                <lookat origin="6,6,3" target="0,0,0.3" up="0,0,1"/>
            </transform>
            <float name="fov" value="60"/>

            <sampler type="ldsampler">
                <integer name="sampleCount" value="256"/>
            </sampler>
            <film type="ldrfilm">
                <integer name="width" value="2400"/>
                <integer name="height" value="480"/>
                <rfilter type="gaussian"/>
                <boolean name="banner" value="false"/>
            </film>
        </sensor>

        <bsdf type="roughplastic" id="surfaceMaterial">
            <float name="alpha" value="0.1"/>
            <float name="intIOR" value="1.46"/>
            <rgb name="diffuseReflectance" value="0.63,0.61,0.58"/> <!-- default 0.5 -->
        </bsdf>

        <shape type="rectangle">
            <transform name="toWorld">
                <lookat origin="0,-11,10" target="0,0,0" /> 
                <scale x="0.5" y="0.5" z="0.5" />
            </transform>
            <emitter type="area">
                <spectrum name="radiance" value="40"/>
            </emitter> 
        </shape>

    """

xml_ball_segment = """
        <shape type="sphere">
            <float name="radius" value="0.025"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="roughplastic">
                <rgb name="diffuseReflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

xml_tail = """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="20" y="20" z="1"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
        </shape>

        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="20" y="20" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="3,3,3"/>
            </emitter>
        </shape>
    </scene>
    """

PALETTE = [
    (25, 95, 235),
    # (8, 30, 74) # original ,
    (255, 102, 99),
    (25, 95, 74),
    (230, 194, 41),
    (241, 113, 5),
]


def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


DEFAULT_COLOR = colormap(8 / 255, 30 / 255, 74 / 255)


def process_single(
    points: np.ndarray, port: int, is_rotated: bool
) -> np.ndarray:
    points = standardize_bbox(points, 2048)

    if not is_rotated:
        points = points[:, [2, 0, 1]]
        points[:, 0] *= -1
    else:
        points[:, 1] *= -1
        points = points[:, [1, 0, 2]]

    points[:, 2] += 0.0125
    xml_segments = [xml_head]

    for i in range(points.shape[0]):
        # color = colormap(
        #     points[i, 0] + 0.5, points[i, 1] + 0.5, points[i, 2] + 0.5 - 0.0125
        # )
        color = colormap(
            # 229/255,194/255,152/255
            # 184 / 255, 151 / 255, 120 / 255
            8 / 255,
            30 / 255,
            74 / 255,
        )
        xml_segments.append(
            xml_ball_segment.format(
                points[i, 0], points[i, 1], points[i, 2], *color
            )
        )
    xml_segments.append(xml_tail)

    xml_content = str.join("", xml_segments)
    result = requests.post(f"http://localhost:{port}/render", data=xml_content)
    data = json.loads(result.content)
    an_img = decode_image(data)
    return an_img


def process_batch(
    points: np.ndarray, port: int, is_rotated: bool
) -> t.Iterator[np.ndarray]:
    for sample in points[:100]:
        yield process_single(sample, port, is_rotated)


def process_scene(
    points: np.ndarray, port: int, is_rotated: bool
) -> np.ndarray:
    xml_segments = [xml_wide_head]
    per_row = 10
    step = 0.8

    for i, shape in enumerate(tqdm.tqdm(points)):
        shape = standardize_bbox(shape, 2048)

        if not is_rotated:
            shape = shape[:, [2, 0, 1]]
            shape[:, 0] *= -1
        else:
            shape[:, 1] *= -1
            shape = shape[:, [1, 0, 2]]

        shape[:, 2] += 0.0125

        aux_shift = np.array(
            [
                float(i // per_row % 2 == 1) * (-step / 2),
                float(i // per_row % 2 == 1) * (step / 2),
                0,
            ]
        )

        starting_point_shift_vec = np.array(
            [-(i // per_row * step), -(i // per_row * step), 0]
        )

        shift_vec = np.array(
            [
                (i % per_row - per_row // 2) * step,
                -(i % per_row - per_row // 2) * step,
                0,
            ]
        )

        combined_shift = aux_shift + starting_point_shift_vec + shift_vec

        shifted_points = shape + combined_shift

        for point in shifted_points:
            color = np.array(PALETTE[i // per_row]) / 255
            xml_segments.append(
                xml_ball_segment.format(point[0], point[1], point[2], *color)
            )
    xml_segments.append(xml_tail)

    xml_content = str.join("", xml_segments)
    result = requests.post(f"http://localhost:{port}/render", data=xml_content)
    data = json.loads(result.content)
    an_img = decode_image(data)
    return an_img


def visualize(
    path: str,
    out_image_path: str,
    is_batch: bool,
    is_torch: bool,
    port: int,
    is_rotated: bool,
    is_scene: bool,
):
    if is_torch:
        points = torch.load(path).detach().cpu().numpy()
    else:
        points = np.load(path)

    if is_scene:
        img = process_scene(points, port, is_rotated)
        cv2.imwrite(out_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    elif is_batch:
        if not os.path.exists(out_image_path):
            os.makedirs(out_image_path)
        pbar = tqdm.tqdm(total=len(points))
        for i, img in enumerate(process_batch(points, port, is_rotated)):
            cv2.imwrite(
                os.path.join(out_image_path, f"{i}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
            pbar.update(1)
        pbar.close()
    else:
        img = process_single(points, port, is_rotated)
        cv2.imwrite(out_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path", help="Path to file from PointFlow dataset"
    )

    parser.add_argument("out", help="Path to image file of rendered points")
    parser.add_argument(
        "--torch",
        action="store_true",
        default=False,
        help="Whether the original tensor is torch.Tensor",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Whether data is batched. If true, that 'out' should folder name",
    )
    parser.add_argument(
        "--rotated",
        action="store_true",
        default=False,
        help="Whether chairs are already rotated",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port of the rendering service"
    )
    parser.add_argument(
        "--scene",
        action="store_true",
        default=False,
        help="Whether render it as a scene",
    )
    args = parser.parse_args()

    visualize(
        args.file_path,
        args.out,
        args.batch,
        args.torch,
        args.port,
        args.rotated,
        args.scene,
    )


if __name__ == "__main__":
    main()
