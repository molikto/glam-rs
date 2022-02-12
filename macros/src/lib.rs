
use proc_macro::TokenStream;


#[proc_macro]
pub fn extract_scalar_type(token: TokenStream) -> TokenStream {
  let width : String = token.to_string().chars().filter(|n| { n.is_ascii_digit()}).collect();
  let typ: String = token.to_string().chars().filter(|n| { n.is_ascii_alphabetic()}).collect();
  if typ == "i" {
    format!("\"OpTypeInt {} 1\"", width).parse().unwrap()
  } else if typ == "u" {
    format!("\"OpTypeInt {} 0\"", width).parse().unwrap()
  } else {
    format!("\"OpTypeFloat {}\"", width).parse().unwrap()
  }
}

#[proc_macro]
pub fn extract_scalar_mul(token: TokenStream) -> TokenStream {
  let typ: String = token.to_string().chars().filter(|n| { n.is_ascii_alphabetic()}).collect();
  let typ = if typ == "i" {
      "OpIMul"
  } else if typ == "u" {
      "OpIMul"
  } else {
      "OpFMul"
  };
  format!("\"{}\"", typ).parse().unwrap()
}

#[proc_macro]
pub fn extract_scalar_add(token: TokenStream) -> TokenStream {
  let typ: String = token.to_string().chars().filter(|n| { n.is_ascii_alphabetic()}).collect();
  let typ = if typ == "i" {
      "OpIAdd"
  } else if typ == "u" {
      "OpIAdd"
  } else {
      "OpFAdd"
  };
  format!("\"{}\"", typ).parse().unwrap()
}

#[proc_macro]
pub fn extract_scalar_sub(token: TokenStream) -> TokenStream {
  let typ: String = token.to_string().chars().filter(|n| {n.is_ascii_alphabetic()}).collect();
  let typ = if typ == "i" {
      "OpISub"
  } else if typ == "u" {
      "OpISub"
  } else {
      "OpFSub"
  };
  format!("\"{}\"", typ).parse().unwrap()
}


#[proc_macro]
pub fn extract_vec_size(token: TokenStream) -> TokenStream {
    let str : String = token.to_string().chars().filter(|n| { n.is_ascii_digit()}).collect();
    format!("\"{}\"", str).parse().unwrap()
}


#[proc_macro]
pub fn extract__to_S(token: TokenStream) -> TokenStream {
    let str : String = token.to_string();
    let str = if str.as_str() == "f32" {
        "OpConvertFToS"
    } else if str.as_str() == "f64" {
        "OpConvertFToS"
    } else if str.as_str() == "u32" {
        "OpSatConvertUToS"
    } else {
      panic!();
    };
    format!("\"{}\"", str).parse().unwrap()
}



#[proc_macro]
pub fn extract__to_F(token: TokenStream) -> TokenStream {
    let str : String = token.to_string();
    let str = if str.as_str() == "f64" {
        "OpFConvert"
    } else if str.as_str() == "u64" {
        "OpConvertUToF"
    } else if str.as_str() == "u32" {
        "OpConvertUToF"
    } else if str.as_str() == "i64" {
        "OpConvertSToF"
    } else if str.as_str() == "i32" {
        "OpConvertSToF"
    } else {
      panic!();
    };
    format!("\"{}\"", str).parse().unwrap()
}