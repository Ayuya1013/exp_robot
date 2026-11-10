import mujoco

print("MuJoCo Version:", mujoco.__version__)

xml_str = """
<mujoco>
  <worldbody>
    <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml_str)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)

print("動作確認成功！ 球体のZ座標:", data.qpos[2])