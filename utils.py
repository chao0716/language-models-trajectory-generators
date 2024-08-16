import pybullet as p
import numpy as np
from openai import OpenAI
import sys
import base64
from io import BytesIO
from PIL import Image
from gtts import gTTS
import time, vlc, os
import pyaudio
import wave
import keyboard
import pybullet as p
import pybullet_data


def text_to_speech(text):
    client = OpenAI()
    speech_file_path = "speech.mp3"
    response = client.audio.speech.create(
      model="tts-1",
      voice="alloy",
      input= text
    )
    response.stream_to_file(speech_file_path)

def audio_to_text(file):
    client = OpenAI()
    audio_file= open(file, "rb")
    translation = client.audio.translations.create(
      model="whisper-1", 
      file=audio_file
    )
    return translation.text

def record_audio(output_filename):
    # Audio settings
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1  # Mono channel
    fs = 44100  # Record at 44100 samples per second

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print("Press 'R' to start recording and release to stop...")

    # Wait for the 'R' key to be pressed
    keyboard.wait('R')

    # Start recording
    print("Recording...")
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    frames = []

    # Continue recording while the 'R' key is pressed
    while keyboard.is_pressed('R'):
        data = stream.read(chunk)
        frames.append(data)

    # Stop recording
    print("Stopped recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as {output_filename}")

def text_to_audio_and_play(text):
    tts = gTTS(text=text, lang='zh-cn')
    audio_file = "temp_audio.mp3"
    tts.save(audio_file)
    player = vlc.MediaPlayer(audio_file)
    # Play the sound file
    player.play()
    # Wait until the sound file finishes playing
    while player.get_state() != vlc.State.Ended:
        time.sleep(1)
    os.remove(audio_file)

def encode_image_to_base64(img_array):
    img_array = np.asarray(img_array, dtype=np.uint8)
    image = Image.fromarray(img_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_str

def get_chatgpt_output(model, new_prompt, messages, role, file=sys.stdout):

    print(role + ":", file=file)
    print(new_prompt, file=file)
    messages.append({"role":role, "content":new_prompt})

    client = OpenAI()

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
        stream=True
    )

    print("assistant:", file=file)

    new_output = ""

    for chunk in completion:
        chunk_content = chunk.choices[0].delta.content
        finish_reason = chunk.choices[0].finish_reason
        if chunk_content is not None:
            print(chunk_content, end="", file=file)
            new_output += chunk_content
        else:
            print("finish_reason:", finish_reason, file=file)

    messages.append({"role":"assistant", "content":new_output})

    return messages



def render_camera_in_sim():

    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    camera_fov = 55
    camera_position = [0, 0, 0.5]  # Camera position (above the blocks)
    target_position = [0, 0, 0]  # Camera looks at the center
    up_vector = [0, 1, 0]

    CAMERA_FAR = 1
    CAMERA_NEAR = 0.15
    HFOV_VFOV = IMAGE_WIDTH/IMAGE_HEIGHT

    view_matrix = p.computeViewMatrix(cameraEyePosition=camera_position,
                                      cameraTargetPosition=target_position,
                                      cameraUpVector=up_vector)

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=camera_fov, aspect=HFOV_VFOV, nearVal=CAMERA_NEAR, farVal=CAMERA_FAR)

    _, _, rgb_image, depth_image, mask = p.getCameraImage(
        width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    # Convert depth image
    depth_image = np.array(depth_image).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

    # Convert depth buffer values to actual depth
    z_depth = CAMERA_FAR * CAMERA_NEAR / \
        (CAMERA_FAR - (CAMERA_FAR - CAMERA_NEAR) * depth_image)

    rgb_image = np.array(rgb_image).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 4)[
        :, :, :3]  # Remove alpha channel

    # Get camera intrinsic parameters
    fx = IMAGE_WIDTH / (2.0 * np.tan(camera_fov * np.pi / 360.0))
    fy = fx
    cx = IMAGE_WIDTH / 2.0
    cy = IMAGE_HEIGHT / 2.0

    # Calculate the 3D points in the camera frame
    x = np.linspace(0, IMAGE_WIDTH - 1, IMAGE_WIDTH)
    y = np.linspace(0, IMAGE_HEIGHT - 1, IMAGE_HEIGHT)
    x, y = np.meshgrid(x, y)

    X = (x - cx) * z_depth / fx
    Y = (y - cy) * z_depth / fy
    Z = camera_position[2] - z_depth

    # Stack the XYZ coordinates
    camera_coordinates = np.stack((X, Y, Z), axis=-1)

    return rgb_image, camera_coordinates

def build_sim():
    # Initialize PyBullet simulation
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    
    # Set gravity
    p.setGravity(0, 0, -9.81)
    
    # Load plane and set the environment
    plane_id = p.loadURDF("plane.urdf")
    
    # Create two cuboids with different colors (3x6x3 cm)
    block1_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.015, 0.015])
    block2_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.015, 0.015])
    
    block1_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.015, 0.015], rgbaColor=[1, 0, 0, 1])  # Red
    block2_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.015, 0.015], rgbaColor=[0, 0, 1, 1])  # Blue
    
    # Create block1 parallel to X-axis (no rotation needed)
    block1_id = p.createMultiBody(baseMass=1.0, 
                                  baseCollisionShapeIndex=block1_collision_shape, 
                                  baseVisualShapeIndex=block1_visual_shape, 
                                  basePosition=[0, 0, 0],
                                  baseOrientation=[0, 0, 0, 1])  # No rotation, default orientation
    
    # Create block2 parallel to Y-axis (90 degrees rotation around Z-axis)
    block2_orientation = p.getQuaternionFromEuler([0, 0, np.pi/2])  # Rotate 90 degrees around Z-axis
    block2_id = p.createMultiBody(baseMass=1.0, 
                                  baseCollisionShapeIndex=block2_collision_shape, 
                                  baseVisualShapeIndex=block2_visual_shape, 
                                  basePosition=[0.15, 0, 0],
                                  baseOrientation=block2_orientation)
