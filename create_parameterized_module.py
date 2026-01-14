import os

def create_parameterized_arm(name, l1, l2, w, tip_angle, filename="arm.urdf"):
    """
    l1, l2: 連桿長度
    w: 厚度 (Width/Thickness)
    tip_angle: 指尖相對於第二連桿的角度 (弧度)
    """
    density = 500  # 假設材料密度 (kg/m^3)
    
    # 計算質量與慣量 (簡化為長方體)
    def get_inertial(m, l, w):
        ixx = (1/12) * m * (l**2 + w**2)
        return f'<mass value="{m:.4f}"/><inertia ixx="{ixx:.6f}" ixy="0" ixz="0" iyy="{ixx:.6f}" iyz="0" izz="{ixx:.6f}"/>'

    m1 = density * (l1 * w * w)
    m2 = density * (l2 * w * w)
    m_tip = density * (0.05 * w * w) # 假設指尖長度固定為 0.05

    urdf_content = f"""
    <robot name="{name}">
      <link name="base_link">
        <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>
      </link>

      <joint name="joint1" type="revolute">
        <parent link="base_link"/><child link="link1"/>
        <axis xyz="0 0 1"/><origin xyz="0 0 0"/>
        <limit effort="100" velocity="10" lower="-3.14" upper="3.14"/>
      </joint>

      <link name="link1">
        <visual><origin xyz="0 {l1/2} 0"/><geometry><box size="{w} {l1} {w}"/></geometry></visual>
        <collision><origin xyz="0 {l1/2} 0"/><geometry><box size="{w} {l1} {w}"/></geometry></collision>
        <inertial>{get_inertial(m1, l1, w)}</inertial>
      </link>

      <joint name="joint2" type="revolute">
        <parent link="link1"/><child link="link2"/>
        <axis xyz="0 0 1"/><origin xyz="0 {l1} 0"/>
        <limit effort="100" velocity="10" lower="-3.14" upper="3.14"/>
      </joint>

      <link name="link2">
        <visual><origin xyz="0 {l2/2} 0"/><geometry><box size="{w} {l2} {w}"/></geometry></visual>
        <collision><origin xyz="0 {l2/2} 0"/><geometry><box size="{w} {l2} {w}"/></geometry></collision>
        <inertial>{get_inertial(m2, l2, w)}</inertial>
      </link>

      <joint name="joint_tip" type="fixed">
        <parent link="link2"/><child link="fingertip"/>
        <origin xyz="0 {l2} 0" rpy="0 0 {tip_angle}"/>
      </joint>

      <link name="fingertip">
        <visual><origin xyz="0 0.025 0"/><geometry><box size="{w*1.2} 0.05 {w}"/></geometry></visual>
        <collision><origin xyz="0 0.025 0"/><geometry><box size="{w*1.2} 0.05 {w}"/></geometry></collision>
        <inertial>{get_inertial(m_tip, 0.05, w)}</inertial>
      </link>
    </robot>
    """
    with open(filename, "w") as f:
        f.write(urdf_content)
    return filename