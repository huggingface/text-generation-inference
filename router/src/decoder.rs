use std::mem::take;
use tokenizers::DecoderWrapper::{BPE, ByteLevel, Metaspace, WordPiece, CTC};
use tokenizers::{Error, Tokenizer};
use unicode_segmentation::UnicodeSegmentation;
use crate::infer::InferError::DetokenizationError;
use crate::infer::InferError;


pub(crate) struct Decoder {
    tokenizer: Tokenizer,
    single_tok_id: u32,
    single_tok: String,
    skip_special_toks: bool,
    pub(crate) seq2seq: bool,
    pub(crate) eos_token_id: u32,
}

impl Decoder {
    pub(crate) fn new(
        tokenizer: Tokenizer, seq2seq: bool, eos_token_id: u32, skip_special_toks: bool,
    ) -> Decoder {
        let prefix_id = *tokenizer.encode("A", false)
            .expect("Tokenizer setup error").get_ids().first().unwrap();
        Decoder {
            single_tok_id: prefix_id,
            single_tok: tokenizer.decode(vec![prefix_id], false).unwrap(),
            tokenizer,
            seq2seq,
            eos_token_id,
            skip_special_toks,
        }
    }

    fn decode_full(&self, ids: Vec<u32>) -> Result<String, InferError> {
        self.tokenizer.decode(ids, self.skip_special_toks).map_err(Error::into)
    }

    pub(crate) fn id_to_token(&self, id: u32) -> String {
        self.tokenizer.id_to_token(id).unwrap_or_else(String::new)
    }

    pub(crate) fn decode(
        &self, mut ids: Vec<u32>, first: bool, last: bool,
    ) -> Result<String, InferError> {
        let decoder = self.tokenizer.get_decoder();
        if (first && self.seq2seq) || (last && matches![decoder, Some(BPE(_))])
            || matches![decoder, Some(ByteLevel(_) | CTC(_))] {
            // In these cases we don't need to do anything special for "continuation"
            let mut text = self.decode_full(ids)?;
            text.truncate(text.trim_end_matches('�').len()); // Avoid add'l allocation
            return Ok(text)
        }
        // How we handle continuation depends on the specific decoder's behaviour,
        // see each one's implementation of decode_chain in the tokenizers library.
        match self.tokenizer.get_decoder() {
            Some(Metaspace(_) | WordPiece(_)) => {
                // For these, the first token in the sequence is treated differently,
                // so we add and then strip a placeholder token.
                ids.insert(0, self.single_tok_id);
                let result = self.decode_full(ids)?;
                Ok(result.strip_prefix(&self.single_tok)
                    .ok_or_else(|| DetokenizationError("Unexpected".into()))
                    ?.to_string())
            },
            Some(BPE(_)) => {
                ids.push(self.single_tok_id);
                let result = self.decode_full(ids)?;
                Ok(result.strip_suffix(&self.single_tok)
                    .ok_or_else(|| DetokenizationError("Unexpected".into()))
                    ?.to_string())
            },
            None => {
                // Just prepend a space
                Ok(format!(" {}", self.decode_full(ids)?))
            },
            _ => {
                Err(DetokenizationError("Unsupported tokenizer type".to_string()))
            }
        }
    }
}

#[derive(Debug)]
pub(crate) enum IncrementalDecoderWrapper {
    ByteLevel(IncrementalBLDecoder), // For ByteLevel
    FirstDiff(IncrementalFirstDiffDecoder), // For Metaspace, WordPiece, None
    LastDiff(IncrementalLastDiffDecoder), // For BPE
    DeDup(IncrementalDeDupDecoder), // For CTE
}

impl IncrementalDecoderWrapper {
    pub(crate) fn for_decoder(decoder: &Decoder, is_start: bool) -> Self {
        match decoder.tokenizer.get_decoder() {
            Some(ByteLevel(_)) => Self::ByteLevel(IncrementalBLDecoder{
                id_buffer: vec![], str_buffer: String::new(), output: String::new(),
            }),
            Some(BPE(_)) => Self::LastDiff(IncrementalLastDiffDecoder{
                output: String::new(), next_id: None,
            }),
            Some(CTC(_)) => Self::DeDup(IncrementalDeDupDecoder{
                output: String::new(), last_id: None,
            }),
            Some(Metaspace(_) | WordPiece(_)) | None | _ => Self::FirstDiff(
                IncrementalFirstDiffDecoder{
                    output: String::new(), first: is_start,
                }
            ),
        }
    }
}

impl IncrementalDecoder for IncrementalDecoderWrapper {
    fn next(&mut self, token: u32, decoder: &Decoder) -> Result<String, InferError> {
        match self {
            Self::ByteLevel(d) => d.next(token, decoder),
            Self::FirstDiff(d) => d.next(token, decoder),
            Self::LastDiff(d) => d.next(token, decoder),
            Self::DeDup(d) => d.next(token, decoder),
        }
    }

    fn flush(&mut self, decoder: &Decoder) -> Result<String, InferError> {
        match self {
            Self::ByteLevel(d) => d.flush(decoder),
            Self::FirstDiff(d) => d.flush(decoder),
            Self::LastDiff(d) => d.flush(decoder),
            Self::DeDup(d) => d.flush(decoder),
        }
    }

    fn output(&self) -> &str {
        match self {
            Self::ByteLevel(d) => d.output(),
            Self::FirstDiff(d) => d.output(),
            Self::LastDiff(d) => d.output(),
            Self::DeDup(d) => d.output(),
        }
    }

    fn into_string(self) -> String {
        match self {
            Self::ByteLevel(d) => d.into_string(),
            Self::FirstDiff(d) => d.into_string(),
            Self::LastDiff(d) => d.into_string(),
            Self::DeDup(d) => d.into_string(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct IncrementalFirstDiffDecoder {
    output: String,
    first: bool,
}

impl IncrementalDecoder for IncrementalFirstDiffDecoder {
    fn next(&mut self, token: u32, decoder: &Decoder) -> Result<String, InferError> {
        let text = decoder.decode(vec![token], self.first, false)?;
        self.first = false;
        self.output += &text;
        Ok(text)
    }

    fn output(&self) -> &str {
        &*self.output
    }
    fn into_string(self) -> String {
        self.output
    }
}

#[derive(Debug)]
pub(crate) struct IncrementalLastDiffDecoder {
    output: String,
    next_id: Option<u32>,
}

impl IncrementalDecoder for IncrementalLastDiffDecoder {
    fn next(&mut self, token: u32, decoder: &Decoder) -> Result<String, InferError> {
        let text = self.next_id.map_or_else(
            || Ok(String::new()),
            |id| decoder.decode(vec![id], true, false)
        )?;
        self.next_id = Some(token);
        self.output += &text;
        Ok(text)
    }

    fn flush(&mut self, decoder: &Decoder) -> Result<String, InferError> {
        let text = self.next_id.map_or_else(
            || Ok(String::new()),
            |id| decoder.decode_full(vec![id])
        )?;
        self.next_id = None;
        self.output += &text;
        Ok(text)
    }

    fn output(&self) -> &str {
        &*self.output
    }
    fn into_string(self) -> String {
        self.output
    }
}

#[derive(Debug)]
pub(crate) struct IncrementalDeDupDecoder {
    output: String,
    last_id: Option<u32>,
}

impl IncrementalDecoder for IncrementalDeDupDecoder {
    fn next(&mut self, token: u32, decoder: &Decoder) -> Result<String, InferError> {
        if self.last_id.map(|id| id == token).unwrap_or(false) {
            return Ok(String::new())
        }
        self.last_id = Some(token);
        let text = decoder.decode_full(vec![token])?;
        self.output += &text;
        Ok(text)
    }

    fn output(&self) -> &str {
        &*self.output
    }
    fn into_string(self) -> String {
        self.output
    }
}


#[derive(Debug)]
pub(crate) struct IncrementalBLDecoder {
    id_buffer: Vec<u32>,
    str_buffer: String,
    output: String,
}


impl IncrementalDecoder for IncrementalBLDecoder {
    fn next(&mut self, token: u32, decoder: &Decoder) -> Result<String, InferError> {
        self.id_buffer.push(token);
        let text = decoder.decode_full(self.id_buffer.clone())?;
        // Defer decoding until we have enough bytes for complete UTF-8
        if !text.ends_with('�') {
            self.output.push_str(&*text);
            if self.str_buffer.is_empty() {
                self.str_buffer = text;
            } else {
                self.str_buffer.push_str(&*text);
            }
            self.id_buffer.clear();

            // Ensure that we return full grapheme clusters
            if let Some((idx, _)) = self.str_buffer
                .grapheme_indices(true).next_back() {
                if idx > 0 {
                    return Ok(self.str_buffer.drain(..idx).collect());
                }
            }
        }
        Ok(String::new())
    }
    fn flush(&mut self, decoder: &Decoder) -> Result<String, InferError> {
        if !self.id_buffer.is_empty() {
            let last = decoder.decode_full(self.id_buffer.clone())?;
            let last = last.trim_end_matches('�');
            self.output += last;
            self.str_buffer.push_str(last);
            self.id_buffer.clear();
        }
        Ok(take(&mut self.str_buffer))
    }

    fn output(&self) -> &str {
        &*self.output
    }
    fn into_string(self) -> String {
        self.output
    }
}

pub(crate) trait IncrementalDecoder {
    fn next(&mut self, token: u32, decoder: &Decoder) -> Result<String, InferError>;
    fn flush(&mut self, _decoder: &Decoder) -> Result<String, InferError> {
        Ok(String::new())
    }
    fn output(&self) -> &str;
    fn into_string(self) -> String;
}


impl From<Error> for InferError {
    fn from(err: Error) -> Self {
        DetokenizationError(err.to_string())
    }
}
