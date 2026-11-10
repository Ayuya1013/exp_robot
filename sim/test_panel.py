import time
import mujoco
import mujoco.viewer

# <include> は <mujoco> 直下に配置
test_xml = """
<mujoco model="panel_test_env">
  <include file="control_panel.xml"/>

  <worldbody>
    <light pos="0 -1 2" dir="0 1 -1"/>
    <!-- 壁面 -->
    <geom name="wall" type="box" size="0.5 0.01 0.5" pos="0 0.1 0" rgba="0.8 0.8 0.8 1"/>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(test_xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)