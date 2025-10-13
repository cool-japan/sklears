use super::neural_types::*;

pub trait Optimizer {
    fn step(&mut self, params: &mut Array2<f64>, gradients: &Array2<f64>);
    fn zero_grad(&mut self);
    fn set_learning_rate(&mut self, lr: f64);
    fn get_learning_rate(&self) -> f64;
}

pub struct SGD {
    pub learning_rate: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub dampening: f64,
    pub nesterov: bool,
    pub velocity: Option<Array2<f64>>,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            velocity: None,
        }
    }

    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }

    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut Array2<f64>, gradients: &Array2<f64>) {
        let mut grad = gradients.clone();

        if self.weight_decay != 0.0 {
            grad = &grad + &(params * self.weight_decay);
        }

        if self.momentum != 0.0 {
            if self.velocity.is_none() {
                self.velocity = Some(Array2::zeros(params.raw_dim()));
            }

            let velocity = self.velocity.as_mut().unwrap();

            if self.dampening == 0.0 {
                *velocity = velocity * self.momentum + &grad;
            } else {
                *velocity = velocity * self.momentum + &grad * (1.0 - self.dampening);
            }

            if self.nesterov {
                grad = &grad + velocity * self.momentum;
            } else {
                grad = velocity.clone();
            }
        }

        *params = params - &grad * self.learning_rate;
    }

    fn zero_grad(&mut self) {
        // SGD doesn't accumulate gradients, so this is a no-op
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
    pub m: Option<Array2<f64>>,
    pub v: Option<Array2<f64>>,
    pub v_max: Option<Array2<f64>>,
    pub step_count: usize,
}

impl Adam {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            m: None,
            v: None,
            v_max: None,
            step_count: 0,
        }
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut Array2<f64>, gradients: &Array2<f64>) {
        self.step_count += 1;

        let mut grad = gradients.clone();

        if self.weight_decay != 0.0 {
            grad = &grad + &(params * self.weight_decay);
        }

        if self.m.is_none() {
            self.m = Some(Array2::zeros(params.raw_dim()));
            self.v = Some(Array2::zeros(params.raw_dim()));
            if self.amsgrad {
                self.v_max = Some(Array2::zeros(params.raw_dim()));
            }
        }

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        *m = m * self.beta1 + &grad * (1.0 - self.beta1);
        *v = v * self.beta2 + &(&grad * &grad) * (1.0 - self.beta2);

        let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);

        let step_size = self.learning_rate / bias_correction1;

        let v_corrected = if self.amsgrad {
            let v_max = self.v_max.as_mut().unwrap();
            for i in 0..v.nrows() {
                for j in 0..v.ncols() {
                    v_max[(i, j)] = v_max[(i, j)].max(v[(i, j)]);
                }
            }
            v_max / bias_correction2
        } else {
            v / bias_correction2
        };

        let mut denominator = Array2::zeros(v_corrected.raw_dim());
        for i in 0..denominator.nrows() {
            for j in 0..denominator.ncols() {
                denominator[(i, j)] = v_corrected[(i, j)].sqrt() + self.eps;
            }
        }

        *params = params - &(m / &denominator) * step_size;
    }

    fn zero_grad(&mut self) {
        // Adam doesn't accumulate gradients, so this is a no-op
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

pub struct AdamW {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub m: Option<Array2<f64>>,
    pub v: Option<Array2<f64>>,
    pub step_count: usize,
}

impl AdamW {
    pub fn new(learning_rate: f64, weight_decay: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            m: None,
            v: None,
            step_count: 0,
        }
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut Array2<f64>, gradients: &Array2<f64>) {
        self.step_count += 1;

        if self.m.is_none() {
            self.m = Some(Array2::zeros(params.raw_dim()));
            self.v = Some(Array2::zeros(params.raw_dim()));
        }

        let m = self.m.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();

        *m = m * self.beta1 + gradients * (1.0 - self.beta1);
        *v = v * self.beta2 + &(gradients * gradients) * (1.0 - self.beta2);

        let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);

        let m_corrected = m / bias_correction1;
        let v_corrected = v / bias_correction2;

        let mut denominator = Array2::zeros(v_corrected.raw_dim());
        for i in 0..denominator.nrows() {
            for j in 0..denominator.ncols() {
                denominator[(i, j)] = v_corrected[(i, j)].sqrt() + self.eps;
            }
        }

        let adam_update = &m_corrected / &denominator;
        let weight_decay_update = params * self.weight_decay;

        *params = params - &(adam_update + weight_decay_update) * self.learning_rate;
    }

    fn zero_grad(&mut self) {
        // AdamW doesn't accumulate gradients, so this is a no-op
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

pub struct RMSprop {
    pub learning_rate: f64,
    pub alpha: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub momentum: f64,
    pub centered: bool,
    pub v: Option<Array2<f64>>,
    pub buf: Option<Array2<f64>>,
    pub grad_avg: Option<Array2<f64>>,
}

impl RMSprop {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            v: None,
            buf: None,
            grad_avg: None,
        }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, params: &mut Array2<f64>, gradients: &Array2<f64>) {
        let mut grad = gradients.clone();

        if self.weight_decay != 0.0 {
            grad = &grad + &(params * self.weight_decay);
        }

        if self.v.is_none() {
            self.v = Some(Array2::zeros(params.raw_dim()));
            if self.momentum > 0.0 {
                self.buf = Some(Array2::zeros(params.raw_dim()));
            }
            if self.centered {
                self.grad_avg = Some(Array2::zeros(params.raw_dim()));
            }
        }

        let v = self.v.as_mut().unwrap();
        *v = v * self.alpha + &(&grad * &grad) * (1.0 - self.alpha);

        let avg = if self.centered {
            let grad_avg = self.grad_avg.as_mut().unwrap();
            *grad_avg = grad_avg * self.alpha + &grad * (1.0 - self.alpha);
            v - &(grad_avg * grad_avg)
        } else {
            v.clone()
        };

        let mut denominator = Array2::zeros(avg.raw_dim());
        for i in 0..denominator.nrows() {
            for j in 0..denominator.ncols() {
                denominator[(i, j)] = avg[(i, j)].sqrt() + self.eps;
            }
        }

        if self.momentum > 0.0 {
            let buf = self.buf.as_mut().unwrap();
            *buf = buf * self.momentum + &(&grad / &denominator);
            *params = params - buf * self.learning_rate;
        } else {
            *params = params - &(&grad / &denominator) * self.learning_rate;
        }
    }

    fn zero_grad(&mut self) {
        // RMSprop doesn't accumulate gradients, so this is a no-op
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

pub struct Adagrad {
    pub learning_rate: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub sum_squared_gradients: Option<Array2<f64>>,
}

impl Adagrad {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            eps: 1e-10,
            weight_decay: 0.0,
            sum_squared_gradients: None,
        }
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Adagrad {
    fn step(&mut self, params: &mut Array2<f64>, gradients: &Array2<f64>) {
        let mut grad = gradients.clone();

        if self.weight_decay != 0.0 {
            grad = &grad + &(params * self.weight_decay);
        }

        if self.sum_squared_gradients.is_none() {
            self.sum_squared_gradients = Some(Array2::zeros(params.raw_dim()));
        }

        let sum_sq_grad = self.sum_squared_gradients.as_mut().unwrap();
        *sum_sq_grad = sum_sq_grad + &(&grad * &grad);

        let mut denominator = Array2::zeros(sum_sq_grad.raw_dim());
        for i in 0..denominator.nrows() {
            for j in 0..denominator.ncols() {
                denominator[(i, j)] = sum_sq_grad[(i, j)].sqrt() + self.eps;
            }
        }

        *params = params - &(&grad / &denominator) * self.learning_rate;
    }

    fn zero_grad(&mut self) {
        // Adagrad doesn't accumulate gradients, so this is a no-op
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

pub struct Adadelta {
    pub learning_rate: f64,
    pub rho: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub sum_squared_gradients: Option<Array2<f64>>,
    pub sum_squared_updates: Option<Array2<f64>>,
}

impl Adadelta {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            rho: 0.9,
            eps: 1e-6,
            weight_decay: 0.0,
            sum_squared_gradients: None,
            sum_squared_updates: None,
        }
    }

    pub fn with_rho(mut self, rho: f64) -> Self {
        self.rho = rho;
        self
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Adadelta {
    fn step(&mut self, params: &mut Array2<f64>, gradients: &Array2<f64>) {
        let mut grad = gradients.clone();

        if self.weight_decay != 0.0 {
            grad = &grad + &(params * self.weight_decay);
        }

        if self.sum_squared_gradients.is_none() {
            self.sum_squared_gradients = Some(Array2::zeros(params.raw_dim()));
            self.sum_squared_updates = Some(Array2::zeros(params.raw_dim()));
        }

        let sum_sq_grad = self.sum_squared_gradients.as_mut().unwrap();
        let sum_sq_updates = self.sum_squared_updates.as_mut().unwrap();

        *sum_sq_grad = sum_sq_grad * self.rho + &(&grad * &grad) * (1.0 - self.rho);

        let mut rms_grad = Array2::zeros(sum_sq_grad.raw_dim());
        let mut rms_update = Array2::zeros(sum_sq_updates.raw_dim());

        for i in 0..rms_grad.nrows() {
            for j in 0..rms_grad.ncols() {
                rms_grad[(i, j)] = (sum_sq_grad[(i, j)] + self.eps).sqrt();
                rms_update[(i, j)] = (sum_sq_updates[(i, j)] + self.eps).sqrt();
            }
        }

        let update = &(&rms_update / &rms_grad) * &grad;

        *params = params - &update * self.learning_rate;

        *sum_sq_updates = sum_sq_updates * self.rho + &(&update * &update) * (1.0 - self.rho);
    }

    fn zero_grad(&mut self) {
        // Adadelta doesn't accumulate gradients, so this is a no-op
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

pub struct LBFGS {
    pub learning_rate: f64,
    pub max_iter: usize,
    pub max_eval: Option<usize>,
    pub tolerance_grad: f64,
    pub tolerance_change: f64,
    pub history_size: usize,
    pub line_search_fn: Option<String>,
}

impl LBFGS {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            max_iter: 20,
            max_eval: None,
            tolerance_grad: 1e-7,
            tolerance_change: 1e-9,
            history_size: 100,
            line_search_fn: None,
        }
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_tolerance_grad(mut self, tolerance_grad: f64) -> Self {
        self.tolerance_grad = tolerance_grad;
        self
    }

    pub fn with_tolerance_change(mut self, tolerance_change: f64) -> Self {
        self.tolerance_change = tolerance_change;
        self
    }

    pub fn with_history_size(mut self, history_size: usize) -> Self {
        self.history_size = history_size;
        self
    }
}

impl Optimizer for LBFGS {
    fn step(&mut self, params: &mut Array2<f64>, gradients: &Array2<f64>) {
        *params = params - gradients * self.learning_rate;
    }

    fn zero_grad(&mut self) {
        // LBFGS doesn't accumulate gradients, so this is a no-op
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}

pub fn create_optimizer(optimizer_type: &str, learning_rate: f64) -> Box<dyn Optimizer> {
    match optimizer_type.to_lowercase().as_str() {
        "sgd" => Box::new(SGD::new(learning_rate)),
        "adam" => Box::new(Adam::new(learning_rate)),
        "adamw" => Box::new(AdamW::new(learning_rate, 0.01)),
        "rmsprop" => Box::new(RMSprop::new(learning_rate)),
        "adagrad" => Box::new(Adagrad::new(learning_rate)),
        "adadelta" => Box::new(Adadelta::new(learning_rate)),
        "lbfgs" => Box::new(LBFGS::new(learning_rate)),
        _ => Box::new(SGD::new(learning_rate)),
    }
}

pub struct LearningRateScheduler {
    pub initial_lr: f64,
    pub scheduler_type: SchedulerType,
    pub step_count: usize,
    pub gamma: f64,
    pub step_size: usize,
    pub milestones: Vec<usize>,
    pub t_max: usize,
    pub eta_min: f64,
}

#[derive(Debug, Clone)]
pub enum SchedulerType {
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LinearLR,
    PolynomialLR { power: f64 },
}

impl LearningRateScheduler {
    pub fn new(initial_lr: f64, scheduler_type: SchedulerType) -> Self {
        Self {
            initial_lr,
            scheduler_type,
            step_count: 0,
            gamma: 0.1,
            step_size: 10,
            milestones: vec![30, 80],
            t_max: 100,
            eta_min: 0.0,
        }
    }

    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn with_step_size(mut self, step_size: usize) -> Self {
        self.step_size = step_size;
        self
    }

    pub fn with_milestones(mut self, milestones: Vec<usize>) -> Self {
        self.milestones = milestones;
        self
    }

    pub fn with_t_max(mut self, t_max: usize) -> Self {
        self.t_max = t_max;
        self
    }

    pub fn with_eta_min(mut self, eta_min: f64) -> Self {
        self.eta_min = eta_min;
        self
    }

    pub fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.step_count += 1;
        let new_lr = self.get_lr();
        optimizer.set_learning_rate(new_lr);
    }

    pub fn get_lr(&self) -> f64 {
        match &self.scheduler_type {
            SchedulerType::StepLR => {
                self.initial_lr * self.gamma.powi((self.step_count / self.step_size) as i32)
            },
            SchedulerType::MultiStepLR => {
                let mut lr = self.initial_lr;
                for &milestone in &self.milestones {
                    if self.step_count >= milestone {
                        lr *= self.gamma;
                    }
                }
                lr
            },
            SchedulerType::ExponentialLR => {
                self.initial_lr * self.gamma.powi(self.step_count as i32)
            },
            SchedulerType::CosineAnnealingLR => {
                self.eta_min + (self.initial_lr - self.eta_min) *
                (1.0 + (std::f64::consts::PI * self.step_count as f64 / self.t_max as f64).cos()) / 2.0
            },
            SchedulerType::LinearLR => {
                let progress = self.step_count as f64 / self.t_max as f64;
                self.initial_lr * (1.0 - progress).max(0.0)
            },
            SchedulerType::PolynomialLR { power } => {
                let progress = self.step_count as f64 / self.t_max as f64;
                self.initial_lr * (1.0 - progress).powf(*power).max(0.0)
            },
            SchedulerType::ReduceLROnPlateau => {
                self.initial_lr
            },
        }
    }
}