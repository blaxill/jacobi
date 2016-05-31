# Jacobi solver

This small library computes the [jacobi
iteration](https://en.wikipedia.org/wiki/Jacobi_method) using a given
[stencil](https://en.wikipedia.org/wiki/Stencil_code). 

# Fluid example

![fluid image](https://raw.githubusercontent.com/blaxill/jacobi/master/fluid-example.gif)

An example of using the jacobi method to solve the discrete pressure Poisson
equation that arrises in fluid simulation is given. The fluid simulation example
can be run through Cargo, and the resulting .ppm frames can be assembled with
ffmpeg:

~~~bash
cargo run --release --example fluid

ffmpeg -y -r 60 -f image2 -i images/fluid-frame%d.ppm -vcodec libx264 -pix_fmt yuv420p -crf 1 -threads 0 -bf 0 images/fluid.mp4
~~~

