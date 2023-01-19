use std::f64::consts::E;

pub trait Activation {
    fn apply(&self, v: f64) -> f64;
    fn derivative(&self, v: f64) -> f64;
}

pub struct Binary;
pub struct Linear(f64);
pub struct Sigmoid;
pub struct Tanh;
pub struct ReLU;
pub struct LeakyReLU;
pub struct ParamReLU(f64);
pub struct ELU(f64);

impl Activation for Binary {
    fn apply(&self, v: f64) -> f64 {
        if v < 0.0 { 0.0 } else { 1.0 }
    }
    fn derivative(&self, v: f64) -> f64 {
        0.0
    }
}

impl Activation for Linear {
    fn apply(&self, v: f64) -> f64 {
        self.0 * v
    }
    fn derivative(&self, v: f64) -> f64 {
        1.0
    }
}

// URL: https://dustinstansbury.github.io/theclevermachine/derivation-common-neural-network-activation-functions
impl Activation for Sigmoid {
    fn apply(&self, v: f64) -> f64 {
        1.0 / (1.0 + E.powf(-v))
    }
    fn derivative(&self, v: f64) -> f64 {
        v * (1.0 - v)
    }
}

// URL: https://www.analyticsvidhya.com/blog/2021/04/activation-functions-and-their-derivatives-a-quick-complete-guide/
// t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
// dt=1-t**2
impl Activation for Tanh {
    fn apply(&self, v: f64) -> f64 {
        (2.0 / (1.0 + E.powf(-v))) - 1.0
    }
    fn derivative(&self, v: f64) -> f64 {
        0.0
    }
}

impl Activation for ReLU {
    fn apply(&self, v: f64) -> f64 {
        if v < 0.0 { 0.0 } else { v }
    }
    fn derivative(&self, v: f64) -> f64 {
        match v {
            _ if v < 0.0 => 0.0,
            _ if v > 0.0 => 1.0,
            _ => f64::NAN // TODO: figure out if this should be 0.0
        }
    }
}

impl Activation for LeakyReLU {
    fn apply(&self, v: f64) -> f64 {
        if v < 0.0 { 0.01*v } else { v }
    }
    fn derivative(&self, v: f64) -> f64 {
        match v {
            _ if v < 0.0 => 0.01,
            _ if v > 0.0 => 1.0,
            _ => f64::NAN // TODO: figure out if this should be 0.0
        }
    }
}

impl Activation for ParamReLU {
    fn apply(&self, v: f64) -> f64 {
        if v < 0.0 { self.0*v } else { v }
    }
    fn derivative(&self, v: f64) -> f64 {
        match v {
            _ if v < 0.0 => self.0,
            _ if v > 0.0 => 1.0,
            _ => f64::NAN // TODO: figure out if this should be 0.0
        }
    }
}

impl Activation for ELU {
    fn apply(&self, v: f64) -> f64 {
        if v < 0.0 {
            self.0 * (E.powf(v) - 1.0)
        } else { v }
    }
    fn derivative(&self, v: f64) -> f64 {
        match v {
            _ if v < 0.0 => self.0 * E.powf(v),
            _ if v > 0.0 => 1.0,
            _ => f64::NAN // TODO: figure out if this should be 0.0
        }
    }
}