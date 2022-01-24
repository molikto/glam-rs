
use proc_macro::TokenStream;


#[proc_macro]
pub fn extract_scalar_size(token: TokenStream) -> TokenStream {
  let str : String = token.to_string().chars().filter(|n| { n.is_ascii_digit()}).collect();
  format!("\"{}\"", str).parse().unwrap()
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