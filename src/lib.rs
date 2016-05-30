#![feature(test)]

extern crate nalgebra;
#[cfg(test)]
extern crate test;

use nalgebra::{BaseFloat, Indexable};

#[derive(Clone)]
pub struct StencilElement<N: BaseFloat> {
    pub offset: usize,
    pub value: N,
}

pub fn jacobi_iter<N, V>(id_inverse: N,
                         pos_stencil: &[StencilElement<N>],
                         neg_stencil: &[StencilElement<N>],
                         x: &mut V, b: &V)
    where N: BaseFloat,
          V: Indexable<usize, N> {
    //
    let len = x.shape();

    for i in 0..len {
        let mut sigma: N = N::zero();

        for s in pos_stencil.iter() {
            let idx = i + s.offset;
            if idx >= len { continue; }
            sigma = sigma + s.value * x[idx];
        }

        for s in neg_stencil.iter() {
            if i <= s.offset { continue; }
            let idx = i - s.offset;
            sigma = sigma + s.value * x[idx];
        }

        let x_i = (b[i] - sigma) * id_inverse;
        x[i] = x_i;
    }
}

#[test]
fn jacobi_test() {
    use nalgebra::DVector;
    let stencil = [
        StencilElement{offset:1, value:-1f32},
        StencilElement{offset:10, value:-1f32}
    ];
    let b = DVector::<f32>::new_random(100);
    let mut x = DVector::<f32>::new_zeros(100);

    for _ in 0..1 {
        jacobi_iter(4f32, &stencil, &stencil, &mut x, &b);
    } 

    let x_1 = x.clone();

    for _ in 0..10 {
        jacobi_iter(4f32, &stencil, &stencil, &mut x, &b);
    } 

    for _ in 0..1000 {
        jacobi_iter(4f32, &stencil, &stencil, &mut x, &b);
    } 

    let x_1011 = x.clone();

    for _ in 0..1000 {
        jacobi_iter(4f32, &stencil, &stencil, &mut x, &b);
    } 

    assert!(x_1 != x);
    assert!(x_1011 == x); // Converged.
}

#[bench]
fn bench_safe (bench: &mut test::Bencher) { 
    use nalgebra::DVector;
    let stencil = [
        StencilElement{offset:1, value:-1f32},
        StencilElement{offset:10, value:-1f32}
    ];
    let b = DVector::<f32>::new_random(100);
    let mut x = DVector::<f32>::new_zeros(100);
    bench.iter(|| jacobi_iter(4f32, &stencil, &stencil, &mut x, &b));
}
