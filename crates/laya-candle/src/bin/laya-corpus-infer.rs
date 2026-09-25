#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ic_laya_core::{engine::InferenceBackend, TokenInput};
    use serde::{Deserialize, Serialize};
    #[derive(Deserialize)]
    struct Case { id: String, input: TokenInput }
    #[derive(Deserialize)]
    struct Corpus { cases: Vec<Case> }
    #[derive(Serialize)]
    struct Output { id: String, logits: Vec<f32> }
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: laya-corpus-infer PACK_DIRECTORY CORPUS_JSON".into());
    }
    let mut model = laya_candle::pack::load_directory(std::path::Path::new(&args[1]))?;
    let corpus: Corpus = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let mut results = Vec::with_capacity(corpus.cases.len());
    for case in corpus.cases {
        results.push(Output { id: case.id, logits: model.infer(&case.input)? });
    }
    println!("{}", serde_json::to_string(&results)?);
    Ok(())
}
#[cfg(target_arch = "wasm32")]
fn main() {}
