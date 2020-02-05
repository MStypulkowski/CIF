import argparse
import json
import typing as t
import matplotlib.pyplot as plt

import cv2
import numpy as np
import requests


def decode_image(byte_data: t.List[float]) -> np.ndarray:
    byte_data = np.asarray(byte_data, dtype=np.uint8)[..., np.newaxis]
    img = cv2.imdecode(byte_data, cv2.IMREAD_COLOR)
    return img


def standardize_bbox(pcl: np.ndarray, points_per_object: int) -> np.ndarray:
    pt_indices = np.random.choice(
        pcl.shape[0], points_per_object, replace=False
    )
    np.random.shuffle(pt_indices)
    pcl = pcl[:points_per_object]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.0
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
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
                <integer name="width" value="800"/>
                <integer name="height" value="600"/>
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

xml_ball_segment = """
        <shape type="sphere">
            <float name="radius" value="0.025"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

xml_tail = """
        <shape type="rectangle">
            <ref name="bsdf" id="surfaceMaterial"/>
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <translate x="0" y="0" z="-0.5"/>
            </transform>
        </shape>
    
        <shape type="rectangle">
            <transform name="toWorld">
                <scale x="10" y="10" z="1"/>
                <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
            </transform>
            <emitter type="area">
                <rgb name="radiance" value="6,6,6"/>
            </emitter>
        </shape>
    </scene>
    """


def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


xml_segments = [xml_head]


def visualize(path: str):
    points = np.load(path)
    points = standardize_bbox(points, 2048)

    points = points[:, [2, 0, 1]]
    points[:, 0] *= -1
    points[:, 2] += 0.0125

    for i in range(points.shape[0]):
        # color = colormap(
        #     points[i, 0] + 0.5, points[i, 1] + 0.5, points[i, 2] + 0.5 - 0.0125
        # )
        color = colormap(
            # 229/255,194/255,152/255
            184 / 255, 151 / 255, 120 / 255
        )
        xml_segments.append(
            xml_ball_segment.format(
                points[i, 0], points[i, 1], points[i, 2], *color
            )
        )
    xml_segments.append(xml_tail)

    xml_content = str.join("", xml_segments)
    result = requests.post("http://localhost:8000/render", data=xml_content)
    data = json.loads(result.content)
    an_img = decode_image(data)

    plt.figure()
    plt.imshow(an_img)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path", help="Path to file from PointFlow dataset"
    )

    args = parser.parse_args()

    visualize(args.file_path)


if __name__ == "__main__":
    main()
