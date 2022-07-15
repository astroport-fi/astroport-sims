use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyTuple};

const FILE_NAME: &str = "simulation.py";
const MODULE_NAME: &str = "simulation";
const FILE_CONTENT: &str = include_str!("simulation.py");

pub const A_MUL: u128 = 10_000;
pub const FEE_MUL: u128 = 1e10 as u128;
pub const MUL_E18: u128 = 1e18 as u128;

pub struct ConcentratedPairModel {
    gil: GILGuard,
    trader: PyObject,
}

impl ConcentratedPairModel {
    pub fn new_default(
        a: u128,
        gamma: u128,
        balances: Vec<u128>,
        n: u128,
        initial_prices: Vec<u128>,
    ) -> PyResult<Self> {
        Self::internal_new(
            (a, gamma, balances.clone(), n, initial_prices.clone()),
            None,
        )
    }

    pub fn new(
        a: u128,
        gamma: u128,
        balances: Vec<u128>,
        n: u128,
        initial_prices: Vec<u128>,
        mid_fee: f64,
        out_fee: f64,
        fee_gamma: u128,
        adjustment_step: f64,
        ma_half_time: u32,
    ) -> PyResult<Self> {
        let kwargs = vec![
            ("mid_fee", mid_fee),
            ("out_fee", out_fee),
            ("fee_gamma", fee_gamma as f64),
            ("adjustment_step", adjustment_step),
            ("ma_half_time", ma_half_time as f64),
        ];

        Self::internal_new(
            (a, gamma, balances.clone(), n, initial_prices.clone()),
            Some(kwargs),
        )
    }

    fn internal_new(
        args: impl IntoPy<Py<PyTuple>>,
        kwargs: Option<Vec<(&str, f64)>>,
    ) -> PyResult<Self> {
        pyo3::prepare_freethreaded_python();

        let gil = Python::acquire_gil();
        let py = gil.python();
        let sim = PyModule::from_code(py, FILE_CONTENT, FILE_NAME, MODULE_NAME)?;
        let trader_class = sim.getattr("Trader")?;
        let trader = trader_class
            .call(args, kwargs.map(|arg| arg.into_py_dict(py)))?
            .to_object(py);

        Ok(Self { gil, trader })
    }

    pub fn call<'a, D>(&'a self, method_name: &str, args: impl IntoPy<Py<PyTuple>>) -> PyResult<D>
    where
        D: FromPyObject<'a>,
    {
        let py = self.gil.python();
        let res_obj = self.trader.call_method1(py, method_name, args)?;
        res_obj.into_ref(py).extract()
    }

    pub fn get_attr<'a, D>(&'a self, attr: &str) -> PyResult<D>
    where
        D: FromPyObject<'a>,
    {
        let py = self.gil.python();
        let res_obj = self.trader.getattr(py, attr)?;
        res_obj.into_ref(py).extract()
    }

    pub fn get_attr_curve<'a, D>(&'a self, attr: &str) -> PyResult<D>
    where
        D: FromPyObject<'a>,
    {
        let py = self.gil.python();
        let curve = self.trader.getattr(py, "curve")?;
        let res_obj = curve.getattr(py, attr)?;
        res_obj.into_ref(py).extract()
    }

    pub fn call_curve<'a, D>(
        &'a self,
        method_name: &str,
        args: impl IntoPy<Py<PyTuple>>,
    ) -> PyResult<D>
    where
        D: FromPyObject<'a>,
    {
        let py = self.gil.python();
        let curve = self.trader.getattr(py, "curve")?;
        let res_obj = curve.call_method1(py, method_name, args)?;
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

    pub fn call_func<'a, D>(&'a self, func: &str, args: impl IntoPy<Py<PyTuple>>) -> PyResult<D>
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
        let model = ConcentratedPairModel::new_default(
            2000 * A_MUL,
            (1e-4 * MUL_E18 as f64) as u128,
            [500_000 * MUL_E18, 250_000 * MUL_E18].to_vec(),
            2,
            vec![MUL_E18, 2 * MUL_E18], // 1 x X = 2 x Y
        )
        .unwrap();

        let fee: u128 = model.call("fee", ()).unwrap();
        assert_eq!(0.001_f32, fee as f32 / FEE_MUL as f32);

        let price: u128 = model.call("price", (0, 1)).unwrap();
        assert_eq!(2_f32, price as f32 / MUL_E18 as f32);

        // Buy Y tokens for 1 x X tokens
        let y_amount: u128 = model.call("buy", (1 * MUL_E18, 0, 1)).unwrap();
        assert_eq!(0.49949999900193914_f64, y_amount as f64 / MUL_E18 as f64);

        // Sell 1 x Y tokens
        let x_amount: u128 = model.call("sell", (1 * MUL_E18, 0, 1)).unwrap();
        assert_eq!(1.9979999999956721_f64, x_amount as f64 / MUL_E18 as f64);

        let model = ConcentratedPairModel::new(
            2000 * A_MUL,
            (1e-4 * MUL_E18 as f64) as u128,
            [500_000 * MUL_E18, 250_000 * MUL_E18].to_vec(),
            2,
            vec![MUL_E18, 2 * MUL_E18],
            0.1,
            0.2,
            1000000,
            0.00001,
            600u32,
        )
        .unwrap();
    }

    #[test]
    fn test_any_func() {
        let res: u128 = Caller::new()
            .call_func("geometric_mean", (vec![100, 100],))
            .unwrap();
        assert_eq!(100, res);
    }

    #[test]
    fn test_call_curve() {
        let model = ConcentratedPairModel::new_default(
            2000 * A_MUL,
            (1e-4 * MUL_E18 as f64) as u128,
            [500_000 * MUL_E18, 250_000 * MUL_E18].to_vec(),
            2,
            vec![MUL_E18, 2 * MUL_E18], // 1 x X = 2 x Y
        )
        .unwrap();

        let res: u128 = model.call_curve("D", ()).unwrap();
        assert_eq!(res as f32 / MUL_E18 as f32, 1_000_000_f32);

        let offer_amount = 100u128;
        let res: u128 = model
            .call_curve("y", ((500_000 + offer_amount) * MUL_E18, 0, 1))
            .unwrap();
        assert_eq!(res / MUL_E18, 249950);
    }

    #[test]
    fn test_getattr() {
        let model = ConcentratedPairModel::new_default(
            2000 * A_MUL,
            (1e-4 * MUL_E18 as f64) as u128,
            [500_000 * MUL_E18, 250_000 * MUL_E18].to_vec(),
            2,
            vec![MUL_E18, 2 * MUL_E18], // 1 x X = 2 x Y
        )
        .unwrap();

        let res: u128 = model.get_attr("xcp_profit").unwrap();
        assert_eq!(res, 1000000000000000000);

        let res: Vec<u128> = model.get_attr_curve("p").unwrap();
        assert_eq!(res, vec![1000000000000000000, 2000000000000000000]);
    }
}
