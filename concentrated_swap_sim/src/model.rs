use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyTuple};

const FILE_NAME: &str = "simulation.py";
const MODULE_NAME: &str = "simulation";
const FILE_CONTENT: &str = include_str!("simulation.py");

pub const A_MUL: u128 = 10_000;
pub const FEE_MUL: u128 = 1e10 as u128;
pub const MUL_E18: u128 = 1e18 as u128;

pub struct ConcentratedPairModel<'s> {
    gil: GILGuard,
    a: u128,
    gamma: u128,
    d: u128,
    n: u128,
    initial_prices: Vec<u128>,
    kwargs: Vec<(&'s str, f32)>,
}

impl<'s> ConcentratedPairModel<'s> {
    pub fn new_default(a: u128, gamma: u128, d: u128, n: u128, initial_prices: Vec<u128>) -> Self {
        pyo3::prepare_freethreaded_python();

        Self {
            gil: Python::acquire_gil(),
            a,
            gamma,
            d,
            n,
            initial_prices,
            kwargs: vec![],
        }
    }

    pub fn new(
        a: u128,
        gamma: u128,
        d: u128,
        n: u128,
        initial_prices: Vec<u128>,
        mid_fee: f32,
        out_fee: f32,
        fee_gamma: f32,
        adjustment_step: f32,
        ma_half_time: u32,
    ) -> Self {
        pyo3::prepare_freethreaded_python();

        let kwargs = vec![
            ("mid_fee", mid_fee),
            ("out_fee", out_fee),
            ("fee_gamma", fee_gamma),
            ("adjustment_step", adjustment_step),
            ("ma_half_time", ma_half_time as f32),
        ];

        Self {
            gil: Python::acquire_gil(),
            a,
            gamma,
            d,
            n,
            initial_prices,
            kwargs,
        }
    }

    pub fn call<'a, D>(&'a self, method_name: &str, args: impl IntoPy<Py<PyTuple>>) -> PyResult<D>
    where
        D: FromPyObject<'a>,
    {
        let py = self.gil.python();
        let sim = PyModule::from_code(py, FILE_CONTENT, FILE_NAME, MODULE_NAME)?;
        let curve_class = sim.getattr("Trader")?;
        let kwargs = if self.kwargs.is_empty() {
            None
        } else {
            Some(self.kwargs.clone().into_py_dict(py))
        };
        let model = curve_class
            .call(
                (
                    self.a,
                    self.gamma,
                    self.d,
                    self.n,
                    self.initial_prices.clone(),
                ),
                kwargs,
            )?
            .to_object(py);
        let res_obj = model.call_method1(py, method_name, args)?;
        res_obj.into_ref(py).extract()
    }
}

pub struct Caller {
    gil: GILGuard,
}

impl Caller {
    pub fn new() -> Self {
        pyo3::prepare_freethreaded_python();

        Self {
            gil: Python::acquire_gil(),
        }
    }

    fn call_func<'a, D>(&'a self, func: &str, args: impl IntoPy<Py<PyTuple>>) -> PyResult<D>
    where
        D: FromPyObject<'a>,
    {
        let code = PyModule::from_code(self.gil.python(), FILE_CONTENT, FILE_NAME, MODULE_NAME)?;

        let res_obj = code.call_method1(func, args)?;
        res_obj.extract()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // You may use this test as an example of how to use the model.
    #[test]
    fn test_concentrated_pair_model() {
        let d = 1_000_000;
        let model = ConcentratedPairModel::new_default(
            2000 * A_MUL,
            (1e-4 * MUL_E18 as f64) as u128,
            d * MUL_E18,
            2,
            vec![MUL_E18, 2 * MUL_E18], // 1 x X = 2 x Y
        );

        let fee: u128 = model.call("fee", ()).unwrap();
        assert_eq!(0.001_f32, fee as f32 / FEE_MUL as f32);

        let price: u128 = model.call("price", (0, 1)).unwrap();
        assert_eq!(2_f32, price as f32 / MUL_E18 as f32);

        // Buy Y tokens for 1 x X tokens
        let y_amount: u128 = model.call("buy", (1 * MUL_E18, 0, 1)).unwrap();
        assert_eq!(0.49949999900193914_f64, y_amount as f64 / MUL_E18 as f64);

        // Sell 1 x Y tokens
        let x_amount: u128 = model.call("sell", (1 * MUL_E18, 0, 1)).unwrap();
        assert_eq!(1.99799999141555_f64, x_amount as f64 / MUL_E18 as f64);
    }

    #[test]
    fn test_any_func() {
        let res: u128 = Caller::new()
            .call_func("geometric_mean", (vec![100, 100],))
            .unwrap();
        assert_eq!(100, res);
    }
}
