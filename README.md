# WebGPU Path Tracer
## NodeJS WebGPU

Path tracing is a highly accurate 3D rendering algorithm that simulates the physical behavior of light to create photorealistic images. By tracing paths of light rays as they bounce off surfaces, it calculates realistic global illumination, shadows, reflections, and refractions.

This demo utilized WebGPU because of efficient low level control of GPU which is well suited for compute intensive operations like path tracing. The absence of rendering pipeline enables efficiency in WebGPU compared to WebGL.


![Demo GIF](https://github.com/arjunsinghgill/WebGPU-path-tracer/blob/main/WebGPU_Path_Tracer.gif)

To run the demo, clone the repository and execute the command:
> node src.WebGPU_path_tracer.https.js
