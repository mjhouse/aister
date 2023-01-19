use std::f64::consts::E;
use rand::Rng;

struct Network {
    weights: Vec<f64>
}

impl Network {

    pub fn new(size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..size)
            .map(|_| rng.gen())
            .collect();
        Self {
            weights: weights
        }
    }

    pub fn __sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + E.powf(-x))
    }

    pub fn __sigmoid_derivative(&self, x: f64) -> f64 {
        x * (1.0 - x)
    }

    pub fn __sigmoid_derivatives(&self, x: &Vec<f64>) -> Vec<f64> {
        x.iter()
         .map(|&v| self.__sigmoid_derivative(v))
         .collect()
    }

    pub fn __dot_product(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        a.iter()
         .zip(b.iter())
         .map(|(x,y)| x * y)
         .sum()
    }

    fn __transpose<T>(&self, v: &Vec<Vec<T>>) -> Vec<Vec<T>>
    where
        T: Clone,
    {
        (0..v[0].len())
            .map(|i| v
                .iter()
                .map(|inner| inner[i].clone())
                .collect())
            .collect()
    }

    pub fn __calculate_errors(&self, a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
        a.iter()
         .zip(b.iter())
         .map(|(x,y)| x * y)
         .collect()
    }

    pub fn train(&mut self, training_inputs: Vec<Vec<f64>>, training_outputs: Vec<f64>, iterations: u64) {
        for _ in 0..iterations {

            // run the inputs through the network
            let output = training_inputs
                .iter()
                .map(|v| self.think(v))
                .collect::<Vec<f64>>();

            // calculate error for each output
            let error = training_outputs
                .iter()
                .zip(output.iter())
                .map(|(x,y)| x - y)
                .collect::<Vec<f64>>();
            
            // get required values for calculating adjustment
            let difference_values = self.__sigmoid_derivatives(&output);
            let transposed_inputs = self.__transpose(&training_inputs);
            let calculated_errors = self.__calculate_errors(&error,&difference_values);

            // calculate an adjustment for each weight
            let adjustment = transposed_inputs
                .iter()
                .map(|v| self.__dot_product(v,&calculated_errors))
                .collect::<Vec<f64>>();
            
            // adjust the weights by the error
            self.weights = self.weights
                .iter()
                .zip(adjustment.iter())
                .map(|(x,y)| x + y)
                .collect();
        } 
    }

    pub fn think(&self, inputs: &Vec<f64>) -> f64 {
        self.__sigmoid(self.__dot_product(
            &self.weights,
            inputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut network = Network::new(3);
        dbg!(&network.weights);

        let training_set_inputs = vec![
            vec![0.0, 0.0, 1.0], 
            vec![1.0, 1.0, 1.0], 
            vec![1.0, 0.0, 1.0], 
            vec![0.0, 1.0, 1.0]
        ];
        let training_set_outputs = vec![
            0.0, 
            1.0, 
            1.0, 
            0.0
        ];


        network.train(training_set_inputs, training_set_outputs, 10000);
        dbg!(&network.weights);

        let result = network.think(&vec![1.0,0.0,0.0]);
        dbg!(&result);
    }
}
