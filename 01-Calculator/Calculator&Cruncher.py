""" 
Project1: Calculator & Number Cruncher 
Concepts covered: Variables, Data Types, Operators, Functions, Loops, Conditionals, User Input
"""

#-------------------------------------------------
# Section 1: Basic Airthmatic Functions
# Concept: Functions, operators, return values
#-------------------------------------------------

def add(a, b):
    """Returns sum to two numbers."""
    return a + b

def substract(a, b):
    """Returns substraction of two numbers. """
    return a - b

def multiply(a, b):
    """Returns product of two numbers. """
    return a * b

def divide(a, b):
    """Returns division result: handles devide-by-zero."""
    if b == 0:
        return "Error: Cannot divide by zero!"
    return a / b

def modulus(a, b):
    """Returns remainder of a divison."""
    if b == 0:
        return "Error: Cannot devide by zero!"
    return a % b

def power(a, b):
    """Returns a raised to the power of b."""
    return a ** b

#--------------------------------------------------------------
# Section 2: Numner Cruncher Utilities
# Concept: Built-In functions, math module, type conversion 
#--------------------------------------------------------------

import math

def is_even_or_odd(n):
    """Check if a number is even or odd. """
    return "Even" if n % 2 == 0 else "Odd"

def is_prime(n):
    """Checks if the number is prime. """
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def factorial(n):
    """Returns factorial of a number using recursion."""
    if n < 0:
        return "Error: Negative numbers have no factorials."
    if n == 0 or n == 1:
        return 1
    return n * factorial(n -1)

def fibonacci(n):
    """Return Fibonacci sequence up to n terms. """
    if n <= 0:
        return []
    sequence = [0, 1]
    for _ in range(2, n):
        sequence.append(sequence[-1] + sequence[-2])
    return sequence[:n]

def find_factors(n):
    """Returns all factors of a number. """
    return [i for i in range(1, abs(n) + 1) if n % i == 0]

def celsius_to_fahrenheit(c):
    """Convert Celsius to Fahrenheit. """
    return (c * 9/5 + 32)

def fahrenheit_to_celsius(f):
    """Convert Farenheit to Celsius. """
    return (f - 32) * 5/9

#--------------------------------------------------------
# Section 3: History Tracker
# Concept: Lists, Append, Iteration
#--------------------------------------------------------

history = []  # Stores past calculations 

def add_to_history(expression, result):
    """Save calculated history. """
    history.append(f" {expression} = {result}")

def show_history():
    """Dispaly all past calculations. """
    if not history:
        print("No History !!!! ")
    else:
        for idx, record in enumerate(history, 1):
            print(f" {idx}. {record}")

#-------------------------------------------------
# Section 4: Input Handler
# User input, type casting, error handling 
#-------------------------------------------------

def get_number(prompt):
    """Prompt user for a valid number. """
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print(" Invalid input. Please enter a number. ")

#---------------------------------------------------
# Section 5: Main Menu (Interactive Loop)
#Concept: While loop, conditionals, match-case (Python 3.10+)
#---------------------------------------------------

def main():
    print("\n" + "="*45)
    print("CALCULATOR & NUMBER CRUNCHER ")
    print("\n" + "="*45)

    while True:
        print("""
-------------MAIN MENU-----------------
              [1] Basic Calculator
              [2] Even / Odd Number 
              [3] Prime number Checker 
              [4] Factorial Calculator 
              [5] Fibonacci Sequence
              [6] Find Factors
              [7] Temparature converter
              [8] View History
              [0] Exit
--------------------------------------""")
        
        choice = input(" Enter Choice: ").strip()

        #----Basic Calculator------
        if choice == "1":
            a = get_number("Enter First Number : ")
            b = get_number("Enter Second Number : ")
            print(""" Operations:
                  [+] Addition
                  [-] Substraction
                  [*] Multiplication 
                  [/] Division 
                  [%] Modulus 
                  [^] Power """)
            op = input("Choose Operator: ").strip()

            if op == "+":
                result = add(a, b)
                expr = f"{a} + {b}"
            elif op == "-":
                result = substract(a, b)
                expr = f"{a} - {b}"
            elif op == "/":
                result = divide(a, b)
                expr = f"{a} / {b}"
            elif op == "*":
                result = multiply(a, b)
                expr = f"{a} * {b}"
            elif op == "%":
                result = modulus(a, b)
                expr = f"{a} % {b}"
            elif op == "^":
                result = power(a, b)
                expr = f"{a} ^ {b}"
            else:
                print(" Invalid Operator !!!!")
                continue

            print(f"\n Result: {expr} = {result}")
            add_to_history(expr, result)

        # ---- Even / Odd ----
        elif choice == "2":
            n = int(get_number(" Enter an integer: "))
            result = is_even_or_odd(n)
            print(f"\n {n} is {result}")
            add_to_history(f"Even/Odd({n})", result)
        
        #-------- Prime ---------
        elif choice == "3":
            n = int(get_number(" Enter an integer: "))
            result = "Prime" if is_prime(n) else "Not a Prime Number"
            print(f"\n {n} is {result}")
            add_to_history(f"Prime({n})", result)
        
        #--------- Factorial ------------
        elif choice == "4":
            n = int(get_number("Enter an non-negative integer: "))
            result = factorial(n)
            print(f"\n {n}! = {result}")
            add_to_history(f"{n}! is", result)

        #---------- Fibonacci ----------------
        elif choice == "5":
            n = int(get_number(" How many Terms ???"))
            result = fibonacci(n)
            print(f"\n  Fibonacci({n}) is", {result})
            add_to_history(f"Fibonacci({n})", result)
        
        #----------- Factors ------------
        elif choice == "6":
            n = int(get_number("Enter an Integer: "))
            result = find_factors(n)
            print(f"\n Factors of {n} are: {result}")
            add_to_history(f"Factors({n})", result)
        
        #------------ Temperatue Converter ---------------
        elif choice == "7":
            print(" [1] Celsius -> Fahrenheit")
            print(" [2] Fahrenheit -> Celsius ")
            sub = input(" Choose: ").strip()
            if sub == "1":
                c = get_number(" Enter Celsius: ")
                result = celsius_to_fahrenheit(c)
                print(f"\n {c} degree C = {result:.2f} degree F")
                add_to_history(f"{c}degree C to degree F", f"{result:.2f} degree F")
            elif sub == '2':
                f = get_number("Enter Fahrenheit: ")
                result = fahrenheit_to_celsius(f)
                print(f"\n {f} dgree F to degree C",f"{result:.2f} degree C")
            else:
                print(" Invalid Choice !!!!!")
        
        #-------- HISTORY ----------
        elif choice == "8":
            print("\n Calculations History:")
            show_history()
        
        # ------ EXIT ---------
        elif choice == "0":
            print("\n Goodbye!!! \n Keep Crunching Numbers!!!!! \n")
            break
        else:
            print("Invalid Choice. Please try again.....")


#----------------------------
# Entery Point 
#----------------------------
if __name__ == "__main__":
    main()