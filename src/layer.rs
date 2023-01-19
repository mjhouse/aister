
struct Layer<const N: usize> {
    weights: [f64;N]
}

impl<const N: usize> Layer<N> {

    pub fn new() -> Self {
        Layer::<N>{
            weights: [0.0;N]
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_layer_with_size() {
        let layer = Layer::<4>::new();
    }
}