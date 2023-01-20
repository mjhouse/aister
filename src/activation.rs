/*
    ## ReLU Derivatives

    The derivatives of the below ReLU-like activations are supposed to be 
    NaN if the input value IS 0.0, but I've seen a few places online that 
    suggest this isn't great for practical purposes and that we should default
    to returning a non-zero value. 

    ## Information

    Most of the information on the different activation functions and their derivatives 
    was taken from this website:

    https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-their-derivatives-a-quick-complete-guide/
    https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/

    ## Notes

    * Derivatives all make use of `f(x)` in the equations, which is equivelent to `v` here

*/

use std::f64::consts::E;

pub trait Activation {
    
    fn activate(&self, v: &mut [f64]);

    fn derivative(&self, v: &mut [f64]);

}

pub struct Binary;
pub struct Linear(f64);
pub struct Sigmoid;
pub struct Tanh;
pub struct ReLU;
pub struct LeakyReLU;
pub struct ParamReLU(f64);
pub struct ELU(f64);
pub struct Swish;
pub struct SoftMax;

impl Activation for Binary {
    fn activate(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            if *x < 0.0 {
                *x = 0.0;
            } else {
                *x = 1.0;
            }
        }
    }
    fn derivative(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            *x = 0.0;
        }
    }
}

impl Activation for Linear {
    fn activate(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            *x = self.0 * (*x);
        }
    }
    fn derivative(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            *x = self.0;
        }
    }
}

impl Activation for Sigmoid {
    fn activate(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            *x = 1.0 / (1.0 + E.powf(-(*x)));
        }
    }
    fn derivative(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            *x = (*x) * (1.0 - (*x));
        }
    }
}

impl Activation for Tanh {
    fn activate(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            *x = (2.0 / (1.0 + E.powf(-2.0 * (*x)))) - 1.0;
        }
    }
    fn derivative(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            *x = 1.0 - (*x).powf(2.0);
        }
    }
}

impl Activation for ReLU {
    fn activate(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            if (*x) < 0.0 {
                *x = 0.0;
            }
        }
    }
    fn derivative(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            if (*x) < 0.0 {
                *x = 0.0;
            } else { 
                *x = 1.0;
            }
        }
    }
}

impl Activation for LeakyReLU {
    fn activate(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            if (*x) < 0.0 { 
                *x = 0.01 * (*x);
            }
        }
    }
    fn derivative(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            if (*x) < 0.0 {
                *x = 0.01;
            } else { 
                *x = 1.0;
            }
        }
    }
}

impl Activation for ParamReLU {
    fn activate(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            if (*x) < 0.0 { 
                *x = self.0 * (*x);
            }
        }
    }
    fn derivative(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            if (*x) < 0.0 {
                *x = self.0;
            } else { 
                *x = 1.0;
            }
        }
    }
}

impl Activation for ELU {
    fn activate(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            if (*x) < 0.0 { 
                *x = self.0 * (E.powf(*x) - 1.0);
            }
        }
    }
    fn derivative(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            if (*x) < 0.0 {
                *x = self.0 + (*x);
            } else { 
                *x = 1.0;
            }
        }
    }
}

impl Activation for Swish {
    fn activate(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            *x = (*x) * (1.0 / (1.0 + E.powf(-(*x))));
        }
    }
    fn derivative(&self, v: &mut [f64]) {
        for x in v.iter_mut() {
            *x = (*x) / (1.0 - E.powf(-(*x)));
        }
    }
}

impl Activation for SoftMax {
    fn activate(&self, v: &mut [f64]) {
        let max: f64 = v
            .iter()
            .max_by(|a,b| a.total_cmp(b))
            .cloned()
            .unwrap_or(0.0);

        let sum: f64 = v.iter()
            .map(|n| n.exp() - max)
            .sum();

        for x in v.iter_mut() {
            let k = x.exp() - max;
            *x = k / sum;
        }
    }
    fn derivative(&self, v: &mut [f64]) {
        self.activate(v);
        for x in v.iter_mut() {
            *x = (*x) * (1.0 - (*x))
        }
    }
}