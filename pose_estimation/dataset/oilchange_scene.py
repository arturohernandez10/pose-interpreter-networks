#"C:\Program Files\Blender Foundation\Blender 2.83\blender.exe" -b -P c:\Projects\AI\DepthPrediction\pose-interpreter-networks\pose_estimation\render_pose.py -- ../data/OilChangeDataset\meshes\engine_high_quality.stl C:\Users\ARTURO~1\AppData\Local\Temp\tmp9cfayuof\render.png object 640 480 514.53821916 513.98831482 311.53091858 254.08105136 0.5 0.6267916603725309,-0.2400053275257872,1.97505545558372 -0.36661718245977093,-0.5256826836753582,0.6397888591893232,-0.4241695100621411


from math import pi

import bpy

def clear_scene():
    for obj in bpy.data.objects:
        if obj.name == 'Camera':
            obj.select_set(False)
        else:
            obj.select_set(True)
    bpy.ops.object.delete()

def place_lamp():
    lamp_data = bpy.data.lights.new(name='lamp', type='POINT')
    lamp = bpy.data.objects.new(name='lamp', object_data=lamp_data)
    bpy.context.collection.objects.link(lamp)
    lamp.location = (0, 0, 0)

def setup_camera(camera_parameters, camera_scale):
    bpy.data.objects['Camera'].location = (0, 0, 0)
    bpy.data.objects['Camera'].rotation_euler = (0, pi, pi)
    width = camera_scale * camera_parameters['width']
    height = camera_scale * camera_parameters['height']
    f = camera_scale * (camera_parameters['f_x'] + camera_parameters['f_y']) / 2.0
    p_x = camera_scale * camera_parameters['p_x']
    p_y = camera_scale * camera_parameters['p_y']
    camera = bpy.data.cameras['Camera']
    camera.lens = 1
    camera.sensor_width = width / f
    camera.shift_x = 0.5 - p_x / width
    camera.shift_y = (p_y - 0.5 * height) / width

def load_model(model_path):
    bpy.ops.import_mesh.stl(filepath=model_path)
    bpy.context.object.location = (0, 0, 0.5)
    bpy.context.object.rotation_mode = 'QUATERNION'
    bpy.context.object.rotation_quaternion = (1, 0, 0, 0)
    bpy.context.object.data.materials.append(None)

def create_object_material():
    object_material = bpy.data.materials.new(name='object')
    #object_material.metallic = 1
    object_material.shadow_method = 'OPAQUE'

def create_mask_material():
    mask_material = bpy.data.materials.new(name='mask')
    mask_material.shadow_method = 'NONE'
    mask_material.use_sss_translucency = True#.translucency = 1.0

def set_render_layers_output():
    tree = bpy.context.scene.node_tree
    render_layers = tree.nodes['Render Layers']
    composite = tree.nodes['Composite']
    tree.links.new(render_layers.outputs[0], composite.inputs[0])

def init(model_path, camera_parameters, camera_scale=0.5):
    clear_scene()
    bpy.context.scene.render.resolution_x = int(camera_scale * camera_parameters['width'])
    bpy.context.scene.render.resolution_y = int(camera_scale * camera_parameters['height'])
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.use_nodes = True
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    place_lamp()
    setup_camera(camera_parameters, camera_scale)
    load_model(model_path)
    create_object_material()
    create_mask_material()
    set_render_layers_output()

def set_mode_object():
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.image_settings.color_depth = '8'
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.look = 'None'
    bpy.context.scene.view_settings.gamma = 1
    bpy.context.scene.view_settings.exposure = 4
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value[:3] = (0, 0, 0)
    #bpy.context.scene.render.simplify_gpencil_antialiasing = True #use_antialiasing = True
    bpy.context.object.data.materials[0] = bpy.data.materials['object']

def set_mode_mask():
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    bpy.context.scene.render.image_settings.color_depth = '8'
    bpy.context.scene.render.use_antialiasing = False
    bpy.context.object.data.materials[0] = bpy.data.materials['mask']

def set_object_pose(position, orientation):
    bpy.context.object.location = position
    bpy.context.object.rotation_quaternion = orientation

def render(output_path):
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
