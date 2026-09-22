//! Opt-in, inclusive instruction spans. Never count nested spans twice.
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cost {
    pub name: String,
    pub shape: [usize; 3],
    pub instructions: u64,
}
struct Recorder {
    counter: fn() -> u64,
    costs: Vec<Cost>,
}
thread_local! { static RECORDER:RefCell<Option<Recorder>>=const{RefCell::new(None)}; }
pub fn measure<T>(name: &'static str, shape: [usize; 3], f: impl FnOnce() -> T) -> T {
    let counter = RECORDER.with(|r| r.borrow().as_ref().map(|r| r.counter));
    let start = counter.map(|c| c());
    let result = f();
    if let (Some(c), Some(start)) = (counter, start) {
        let instructions = c().saturating_sub(start);
        RECORDER.with(|r| {
            if let Some(r) = r.borrow_mut().as_mut() {
                r.costs.push(Cost {
                    name: name.into(),
                    shape,
                    instructions,
                });
            }
        });
    }
    result
}
pub fn capture<T>(counter: fn() -> u64, f: impl FnOnce() -> T) -> (T, Vec<Cost>) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            RECORDER.with(|r| *r.borrow_mut() = None);
        }
    }
    RECORDER.with(|r| {
        assert!(r.borrow().is_none(), "nested profile capture");
        *r.borrow_mut() = Some(Recorder {
            counter,
            costs: vec![],
        });
    });
    let _reset = Reset;
    let result = f();
    let costs = RECORDER.with(|r| r.borrow_mut().take().unwrap().costs);
    (result, costs)
}
