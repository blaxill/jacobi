extern crate nalgebra;
#[cfg(test)]
extern crate test;

use nalgebra::{BaseFloat, Indexable};

#[derive(Clone)]
pub struct StencilElement<N: BaseFloat> {
    pub offset: isize,
    pub value: N,
}

/// Computes a [jacobi iteration](https://en.wikipedia.org/wiki/Jacobi_method)
/// using a [stencil](https://en.wikipedia.org/wiki/Stencil_code). Reapted
/// applications of this method solve Ax=b, where the matrix A can be
/// represented by a stencil. `id_inverse` is the inverse of the main diagonal
/// element A_ii. `stencil` is some iterable representing the other stencil
/// elements of A.
/// 
/// # Examples
///
/// Using the jacobi method to approximate pressure from the divergence of a
/// velocity field in a 256x256 grid:
///
/// ```
/// let size = 256*256;
/// let stencil = [ 
///     StencilElement{offset:1, value:1f32}, 
///     StencilElement{offset:256, value:1f32}, 
///     StencilElement{offset:-1, value:1f32}, 
///     StencilElement{offset:-256, value:1f32} 
/// ];
///
/// let divergence = DVector::<f32>::new_zeros(size); 
/// let mut pressure_grid = DVector::<f32>::new_zeros(size); 
/// let mut refined_pressure_grid = DVector::<f32>::new_zeros(size);
///
/// for _ in 0..50 { 
///     jacobi_iter(-0.25f32, stencil.iter(), 
///         &pressure_grid, 
///         &mut refined_pressure_grid, 
///         &divergence);
///     std::mem::swap(&mut pressure_grid, &mut refined_pressure_grid);
/// }
///
/// let _ = pressure_grid;
///
/// ```
///
pub fn jacobi_iter<'a, Float, Vector, Stencil>(id_inverse: Float,
                                               stencil: Stencil,
                                               x: &Vector,
                                               x_n: &mut Vector,
                                               b: &Vector)
    where Float: 'static + BaseFloat,
          Vector: Indexable<usize, Float>,
          Stencil: Iterator<Item = &'a StencilElement<Float>>
{
    //
    use std::cmp::{max, min};
    let len = x.shape();

    for i in 0..len {
        unsafe {
            x_n.unsafe_set(i, b.unsafe_at(i) * id_inverse);
        }
    }

    for s in stencil {
        let start = max(-s.offset, 0) as usize;
        let stop = min(len as isize - s.offset, len as isize) as usize;
        let multiplier = s.value * id_inverse;

        for i in start..stop {
            let sample_offset = (i as isize + s.offset) as usize;
            let current = unsafe { x_n.unsafe_at(i) };
            unsafe {
                x_n.unsafe_set(i, current - x.unsafe_at(sample_offset) * multiplier);
            };
        }

    }
}
