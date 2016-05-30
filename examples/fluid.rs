#![feature(alloc_system)]

extern crate nalgebra;
extern crate jacobi;
extern crate alloc_system;

use jacobi::{jacobi_iter, StencilElement};
use nalgebra::{DVector, Vector2};
use std::io::Write;
use std::fs::File;
use std::f32;

pub fn write_ppm<W: Write>(w: &mut W,
                        c: &DVector<f32>,
                        v: &DVector<Vector2<f32>>,
                        width: usize, height: usize) {
    write!(w, "P3\n{} {}\n255\n", width, height).unwrap();

    let f = |v| -> u8 {(v * 255f32) as u8};
    let n = |v: f32| -> u8 {(v * 128f32 + 128f32) as u8};

    for i in 0..height {
        for j in 0..width {
            if cfg!(debug) {
                write!(w, "{} {} {} ",
                       f(c[j + 256*i]),
                       n(v[j + 256*i].x),
                       n(v[j + 256*i].y)).unwrap();
            } else { 
                write!(w, "{} {} {} ",
                       f(c[j + 256*i]),
                       f(c[j + 256*i]),
                       f(c[j + 256*i])).unwrap();
            }
        }
        write!(w, "\n").unwrap();
    }
}

fn main() {
    // 256x256 grid.
    let size = 256*256;
    let stencil = [
        StencilElement{offset:1, value:1f32},
        StencilElement{offset:256, value:1f32}
    ];
    let indexer = |x: usize, y: usize| -> usize { x + 256*y };


    let mut color_grid = DVector::<f32>::new_zeros(size);
    let mut velocity_grid = DVector::<Vector2<f32>>::new_zeros(size);
    let mut next_color_grid = DVector::<f32>::new_zeros(size);
    let mut next_velocity_grid = DVector::<Vector2<f32>>::new_zeros(size);
    let mut divergence = DVector::<f32>::new_zeros(size);
    let mut pressure_grid = DVector::<f32>::new_zeros(size);

    // Simulate 200 timesteps.
    for _ in 0..200 {
        velocity_grid[indexer(127,0)] = Vector2::new(0.0f32, 3.0f32);
        velocity_grid[indexer(128,0)] = Vector2::new(0.0f32, 3.0f32);
        velocity_grid[indexer(126,0)] = Vector2::new(0.0f32, 3.0f32);
        velocity_grid[indexer(129,0)] = Vector2::new(0.0f32, 3.0f32);

        color_grid[indexer(127,0)] = 1f32;
        color_grid[indexer(128,0)] = 1f32;
        color_grid[indexer(126,0)] = 1f32;
        color_grid[indexer(129,0)] = 1f32;

        // 1. Advect color markers and velocity.
        for x in 0..256 {
            for y in 0..256 {
                let delta = -velocity_grid[indexer(x, y)];
                let sample_location = Vector2::new(x as f32, y as f32) + delta;
                let sx = if sample_location.x < 0f32 { 0f32 }
                    else if sample_location.x > 255f32 { 255f32 }
                    else { sample_location.x };
                let sy = if sample_location.y < 0f32 { 0f32 }
                    else if sample_location.y > 255f32 { 255f32 }
                    else { sample_location.y };

                let x_floor = sx.floor();
                let y_floor = sy.floor();
                let x_ceil = sx.ceil();
                let y_ceil = sy.ceil();
                let x_frac = sx - x_floor;
                let y_frac = sy - y_floor;

                let f_indexer = |x, y| { indexer(x as usize, y as usize) };
                
                let c_0 = color_grid[f_indexer(x_floor,y_floor)] * (1f32 - x_frac)
                        + color_grid[f_indexer(x_ceil,y_floor)] * x_frac;

                let c_1 = color_grid[f_indexer(x_floor,y_ceil)] * (1f32 - x_frac)
                        + color_grid[f_indexer(x_ceil,y_ceil)] * x_frac;

                let v_0 = velocity_grid[f_indexer(x_floor,y_floor)] * (1f32 - x_frac)
                        + velocity_grid[f_indexer(x_ceil,y_floor)] * x_frac;

                let v_1 = velocity_grid[f_indexer(x_floor,y_ceil)] * (1f32 - x_frac)
                        + velocity_grid[f_indexer(x_ceil,y_ceil)] * x_frac;

                next_color_grid[indexer(x, y)] = c_0 * (1f32 - y_frac) + c_1 * y_frac;
                next_velocity_grid[indexer(x, y)] = v_0 * (1f32 - y_frac) + v_1 * y_frac;
            }
        }
        std::mem::swap(&mut color_grid, &mut next_color_grid);
        std::mem::swap(&mut velocity_grid, &mut next_velocity_grid);

        // 2. Calculate divergence
        for x in 0..256 {
            for y in 0..256 {
                let vx = match x {
                    0 => velocity_grid[indexer(x+1, y)].x,
                    255 => -velocity_grid[indexer(x-1, y)].x,
                    x => {
                        let px_0 = velocity_grid[indexer(x-1, y)].x;
                        let px_1 = velocity_grid[indexer(x+1, y)].x;
                        (px_1 - px_0) / 2f32
                    },
                };
                let vy = match y {
                    0 => velocity_grid[indexer(x, y+1)].y,
                    255 => -velocity_grid[indexer(x, y-1)].y,
                    y => {
                        let py_0 = velocity_grid[indexer(x, y-1)].y;
                        let py_1 = velocity_grid[indexer(x, y+1)].y;
                        (py_1 - py_0) / 2f32
                    },
                };
                divergence[indexer(x, y)] = vx + vy;
            }
        }

        // 3. Approximate pressure jacobi iterations.
        for _ in 0..50 { 
            jacobi_iter(-0.25f32, &stencil, &stencil,
                        &mut pressure_grid, &divergence);
        }

        // 4. Apply pressure.
        for x in 0..256 {
            for y in 0..256 {
                let vx = match x {
                    0 | 255 => 0.0f32,
                    x => {
                        let px_0 = pressure_grid[indexer(x-1, y)];
                        let px_1 = pressure_grid[indexer(x+1, y)];
                        (px_1 - px_0) / 2.0f32
                    },
                };
                let vy = match y {
                    0 | 255 => 0.0f32,
                    y => {
                        let py_0 = pressure_grid[indexer(x, y-1)];
                        let py_1 = pressure_grid[indexer(x, y+1)];
                        (py_1 - py_0) / 2.0f32
                    },
                };
                velocity_grid[indexer(x, y)] -= Vector2::new(vx, vy);
            }
        }
    }

    let mut file = File::create("out.ppm").unwrap();
    let _ = write_ppm(&mut file, &color_grid, &velocity_grid, 256, 256);
}
