"""
Build jot corpus from all .jot source files + synthetic examples.
Writes to data/jot_corpus.txt
"""

import os
import glob

JOT_DIRS = [
    os.path.expanduser("~/Documents/Code/jot/examples"),
    os.path.expanduser("~/Documents/Code/jot/examples/lib"),
    os.path.expanduser("~/Documents/Code/jot/apps"),
]

SYNTHETIC = """
// Basic arithmetic
let a = 10;
let b = 3;
let sum = a + b;
let diff = a - b;
let prod = a * b;
let quot = a / b;
let rem = a % b;
print sum;
print diff;
print prod;
print quot;
print rem;

// String operations
let greeting = "Hello";
let name = "world";
let msg = greeting + ", " + name + "!";
print msg;

// Boolean logic
let x = true;
let y = false;
if x and !y {
    print "logic works";
}

// Nested functions
fn max(a, b) {
    if a > b {
        return a;
    }
    return b;
}

fn min(a, b) {
    if a < b {
        return a;
    }
    return b;
}

fn clamp(val, lo, hi) {
    return max(lo, min(val, hi));
}

print clamp(15, 0, 10);
print clamp(-5, 0, 10);
print clamp(5, 0, 10);

// Recursive fibonacci
fn fib(n) {
    if n <= 1 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

let i = 0;
while i < 10 {
    print fib(i);
    i = i + 1;
}

// Nested loops
let row = 0;
while row < 5 {
    let col = 0;
    let line = "";
    while col < 5 {
        if col <= row {
            line = line + "*";
        } else {
            line = line + " ";
        }
        col = col + 1;
    }
    print line;
    row = row + 1;
}

// Array manipulation
let arr = [5, 3, 8, 1, 9, 2, 7, 4, 6];
print "Before sort: " + stringify(arr);
let sorted = sort(arr);
print "After sort: " + stringify(sorted);

// String formatting
fn pad_left(s, width) {
    let result = s;
    while len(result) < width {
        result = " " + result;
    }
    return result;
}

fn repeat_char(c, n) {
    let s = "";
    let i = 0;
    while i < n {
        s = s + c;
        i = i + 1;
    }
    return s;
}

print pad_left("42", 5);
print repeat_char("-", 20);

// Object patterns
let config = {
    host: "localhost",
    port: 8080,
    debug: true,
    timeout: 30
};

if config.debug {
    print "Debug mode on " + config.host + ":" + str(config.port);
}

// Data processing
fn sum_array(arr) {
    let total = 0;
    let i = 0;
    while i < len(arr) {
        total = total + arr[i];
        i = i + 1;
    }
    return total;
}

fn average(arr) {
    return sum_array(arr) / len(arr);
}

let scores = [85, 92, 78, 95, 88, 73, 91];
print "Sum: " + str(sum_array(scores));
print "Average: " + str(average(scores));

// Class with methods
class Stack {
    fn init() {
        this.items = [];
    }

    fn push(val) {
        push(this.items, val);
    }

    fn pop() {
        return pop(this.items);
    }

    fn peek() {
        return this.items[len(this.items) - 1];
    }

    fn size() {
        return len(this.items);
    }

    fn is_empty() {
        return len(this.items) == 0;
    }
}

let stack = new Stack();
stack.push(1);
stack.push(2);
stack.push(3);
print "Size: " + str(stack.size());
print "Peek: " + str(stack.peek());
print "Pop: " + str(stack.pop());
print "Size: " + str(stack.size());

// Linked list
class Node {
    fn init(val) {
        this.val = val;
        this.next = null;
    }
}

class LinkedList {
    fn init() {
        this.head = null;
        this.length = 0;
    }

    fn append(val) {
        let node = new Node(val);
        if this.head == null {
            this.head = node;
        } else {
            let cur = this.head;
            while cur.next != null {
                cur = cur.next;
            }
            cur.next = node;
        }
        this.length = this.length + 1;
    }

    fn print_all() {
        let cur = this.head;
        while cur != null {
            print str(cur.val);
            cur = cur.next;
        }
    }
}

let list = new LinkedList();
list.append(10);
list.append(20);
list.append(30);
print "List:";
list.print_all();

// Error handling patterns
fn safe_divide(a, b) {
    if b == 0 {
        return 0;
    }
    return a / b;
}

fn parse_int(s) {
    return int(s);
}

// Range and for loops combined
let evens = [];
for i in range(0, 20) {
    if i % 2 == 0 {
        push(evens, i);
    }
}
print "Evens: " + stringify(evens);

// Closure-like pattern with objects
fn make_counter(start) {
    let state = {count: start};
    fn increment() {
        state.count = state.count + 1;
        return state.count;
    }
    return increment;
}

// Recursive descent
fn gcd(a, b) {
    if b == 0 {
        return a;
    }
    return gcd(b, a % b);
}

fn lcm(a, b) {
    return (a * b) / gcd(a, b);
}

print "gcd(48, 18) = " + str(gcd(48, 18));
print "lcm(4, 6) = " + str(lcm(4, 6));

// Map/filter/reduce patterns
fn square(x) {
    return x * x;
}

fn is_odd(x) {
    return x % 2 != 0;
}

fn add(a, b) {
    return a + b;
}

let nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let squares = map("square", nums);
let odds = filter("is_odd", nums);
let total = reduce("add", nums, 0);

print "Squares: " + stringify(squares);
print "Odds: " + stringify(odds);
print "Total: " + str(total);

// String parsing
fn starts_with(s, prefix) {
    if len(s) < len(prefix) {
        return false;
    }
    let i = 0;
    while i < len(prefix) {
        if s[i] != prefix[i] {
            return false;
        }
        i = i + 1;
    }
    return true;
}

fn ends_with(s, suffix) {
    if len(s) < len(suffix) {
        return false;
    }
    let offset = len(s) - len(suffix);
    let i = 0;
    while i < len(suffix) {
        if s[offset + i] != suffix[i] {
            return false;
        }
        i = i + 1;
    }
    return true;
}

// Bubble sort implementation
fn bubble_sort(arr) {
    let n = len(arr);
    let i = 0;
    while i < n {
        let j = 0;
        while j < n - i - 1 {
            if arr[j] > arr[j + 1] {
                let tmp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = tmp;
            }
            j = j + 1;
        }
        i = i + 1;
    }
    return arr;
}

let unsorted = [64, 34, 25, 12, 22, 11, 90];
print "Sorted: " + stringify(bubble_sort(unsorted));

// Binary search
fn binary_search(arr, target) {
    let lo = 0;
    let hi = len(arr) - 1;
    while lo <= hi {
        let mid = (lo + hi) / 2;
        if arr[mid] == target {
            return mid;
        } else {
            if arr[mid] < target {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
    }
    return -1;
}

let sorted_arr = [1, 3, 5, 7, 9, 11, 13, 15];
print "Found 7 at index: " + str(binary_search(sorted_arr, 7));
print "Found 6 at index: " + str(binary_search(sorted_arr, 6));

// Event-driven pattern
fn on_click(button) {
    if button == "submit" {
        print "Form submitted!";
        return true;
    }
    if button == "cancel" {
        print "Cancelled.";
        return false;
    }
    print "Unknown button: " + button;
    return false;
}

on_click("submit");
on_click("cancel");
on_click("help");

// State machine
class StateMachine {
    fn init() {
        this.state = "idle";
    }

    fn transition(event) {
        if this.state == "idle" {
            if event == "start" {
                this.state = "running";
                return "started";
            }
        }
        if this.state == "running" {
            if event == "pause" {
                this.state = "paused";
                return "paused";
            }
            if event == "stop" {
                this.state = "idle";
                return "stopped";
            }
        }
        if this.state == "paused" {
            if event == "resume" {
                this.state = "running";
                return "resumed";
            }
            if event == "stop" {
                this.state = "idle";
                return "stopped";
            }
        }
        return "invalid transition";
    }

    fn get_state() {
        return this.state;
    }
}

let sm = new StateMachine();
print sm.get_state();
print sm.transition("start");
print sm.get_state();
print sm.transition("pause");
print sm.transition("resume");
print sm.transition("stop");
print sm.get_state();

// Prime sieve
fn is_prime(n) {
    if n < 2 {
        return false;
    }
    let i = 2;
    while i * i <= n {
        if n % i == 0 {
            return false;
        }
        i = i + 1;
    }
    return true;
}

let primes = [];
for n in range(2, 50) {
    if is_prime(n) {
        push(primes, n);
    }
}
print "Primes < 50: " + stringify(primes);

// Matrix as array of arrays
let matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
];

fn matrix_get(m, row, col) {
    return m[row][col];
}

fn matrix_trace(m) {
    let n = len(m);
    let sum = 0;
    let i = 0;
    while i < n {
        sum = sum + m[i][i];
        i = i + 1;
    }
    return sum;
}

print "Trace: " + str(matrix_trace(matrix));

// Temperature converter
fn celsius_to_fahrenheit(c) {
    return c * 9 / 5 + 32;
}

fn fahrenheit_to_celsius(f) {
    return (f - 32) * 5 / 9;
}

for c in [0, 20, 37, 100] {
    print str(c) + "C = " + str(celsius_to_fahrenheit(c)) + "F";
}
"""


def build_corpus():
    parts = []

    # Collect all .jot files
    for d in JOT_DIRS:
        for path in sorted(glob.glob(os.path.join(d, "*.jot"))):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            parts.append(f"// --- {os.path.basename(path)} ---\n{content}\n\n")

    # Add synthetic examples
    parts.append("// --- synthetic.jot ---\n")
    parts.append(SYNTHETIC)

    corpus = "\n".join(parts)

    # Strip non-ASCII to keep vocab clean for char-level modeling
    corpus = corpus.encode("ascii", errors="ignore").decode("ascii")

    out_path = os.path.join(os.path.dirname(__file__), "data", "jot_corpus.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(corpus)

    print(f"Wrote {len(corpus):,} chars to {out_path}")
    print(f"Unique chars: {len(set(corpus))}")
    return corpus


if __name__ == "__main__":
    build_corpus()
