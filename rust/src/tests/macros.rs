// 宏定义
#[macro_export]
macro_rules! push_fields {
        ($row:expr, $record:expr, $($field:ident),*) => {
            $(
                $row.push($record.$field);
            )*
        };
    }

pub use push_fields;
