from setuptools import setup, find_packages

package_name = 'by_your_command'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'openai-whisper>=2023.7.1',
        'openai>=1.0.0',
        'silero-vad>=0.3.1',
        'torch>=1.12.0',
        'python-dotenv>=0.21.0',
        'websockets>=11.0',
        'aiohttp>=3.8.0',
        'pydantic>=2.0',
        'numpy>=1.20.0',
        'PyYAML>=6.0',
    ],
    author='TODO: Maintainer Name',
    author_email='you@todo.todo',
    description='ByYourCommand: voice, camera, and video interaction package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'silero_vad_node = voice_detection.silero_vad_node:main',
            'interaction_node = interactions.interaction_node:main',
            'speech_chunk_recorder = voice_detection.speech_chunk_recorder:main',
            'ros_ai_bridge = ros_ai_bridge.ros_ai_bridge:main',
            'oai_realtime_agent = agents.main:main',
        ],
    },
)
