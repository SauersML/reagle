fn main() {
    let mut x = bitvec::bitvec![u64, bitvec::prelude::Lsb0; 0; 10];
    x.set(1, true);
    println!("{:?}", x);
}
