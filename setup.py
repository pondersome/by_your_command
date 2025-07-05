from setuptools import setup, find_packages

package_name = 'by_your_command'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, 'silero_vad', 'interactions'],
    install_requires=[
        'openai-whisper>=2023.7.1',
        'openai>=0.27.0',
        'silero-vad>=0.3.1',
        'torch>=1.12.0',
        'python-dotenv>=0.21.0',
    ],
    author='TODO: Maintainer Name',
    author_email='you@todo.todo',
    description='ByYourCommand: voice, camera, and video interaction package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'silero_vad_node = silero_vad.silero_vad_node:main',
            'interaction_node = interactions.interaction_node:main',
            'speech_only = silero_vad.speech_only:main',
        ],
    },
)
