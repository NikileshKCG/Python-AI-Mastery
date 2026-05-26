"""
Project 2: Student Grade Manager
Concepts: Lists, Dictionaries, Loops, Conditionals, File I/O (csv),
          String Methods, Sorting, Exception Handling, Functions
"""

import csv
import os
import statistics

#-------------------------------------------------------
#SECTION 1: In-Memory Data Store
#Concept: Dictionary of lists - the core data structure
#
#Structure:
#  Students = {
#       "Raj": {"grades": [85, 90, 78], "subject": "Math"},
#       "Sara": {"grades": [92, 85, 95], "subject": "Science"}
# }
#-------------------------------------------------------

students = {} # Main data store - a dict where each value is also a dict

#-------------------------------------------------------
# SECTION 2: Grade Logic
# Concept: Functions, conditionals, math operations
#-------------------------------------------------------

def calculate_average(grades):
    """Return average of a list of grades."""
    if not grades:
        return 0.0
    return sum(grades) / len(grades)

def get_letter_graade(average):
    """Convert numberic average to letter grade. """
    if average >= 90:
        return "A"
    elif average >= 80:
        return "B"
    elif average >= 70:
        return "C"
    elif average >= 60:
        return "D"
    elif average >= 50:
        return "E"
    else:
        return "F"

def get_grade_remark(letter):
    """Return a remark based on letter grade. """
    remarks = {
        "A": "Excellent !!!",
        "B": "Good Job !!",
        "C": "Average, Keep Pushing !",
        "D": "Below Average Needs imporovement..",
        "E": "Just about passed, scope for a lot of improvement",
        "F": "Failing -- urgent attention needed"
    }
    return remarks.get(letter, "N/A")

#----------------------------------------------------
# SECTION 3: Student CRUD Operations
# Concept: Dict manipulation, list, .append(), input validation
#----------------------------------------------------

def add_student(name, subject, grades):
    """ Add a new student with ther suject and grades. """
    name = name.strip().title()
    if name in students:
        print(f" '{name}' already existis. Use 'Update' to add grades. ")
        return
    students[name] = {
        "subject": subject.strip().title(),
        "grades": grades
    }
    print(f" Student '{name}' added successfully. ")

def update_grades(name, new_grades):
    """Append new grades to an existing student. """
    name = name.strip().title()
    if name not in students:
        print(f" Studnet '{name}' not found. ")
        return
    students[name]["grades"].extend(new_grades) # .extend adds multiple items to the list
    print(f" Grades updated for '{name}'.")

def delete_student(name):
    """Remove a student from the records. """
    name = name.strip().title()
    if name in students:
        del students[name]
        print(f" Student '{name}' is deleted. ")
    else:
        print(f" Sutent '{name}' not found . ")

def view_student(name):
    """Dispaly detailed report for one student."""
    name = name.strip().title()
    if name not in students:
        print(f" Student '{name}' not found. ")
        return
    
    info = students[name]
    grades = info["grades"]
    avg = calculate_average(grades)
    letter = get_letter_graade(avg)
    remark = get_grade_remark(letter)

    print(f"""
          -----------------------------------------------
          |  Student : {name: <25} |
          |  Subject : {info['subject']:<25} |
          |  Grades  : {str(grades):<25} |
          |  Average : {avg:<25.2f} |
          |  Highest : {max(grades):<25} |
          |  Lowest  : {min(grades):<25} |
          |  Grade   : {letter:<25} |
          |  Remark  : {remark:<25}
          -----------------------------------------------
          """)

#------------------------------------------------------------
# SECTION 4: Class-wide Reports
#Concept: Iterating over dicts, string, statistics module
#------------------------------------------------------------

def view_all_students():
    """Dispaly a summary table of all the students. """
    if not students:
        print(" No Students Found !!!! ")
        return
    
    print(f"\n {'Name':,20} {'Subject':<15} {'Avg':<6} {'Grade':>6} {'Remark'}")
    print(" " + "-" * 64)

    # iterate over dictionary items - Key=name, value=info dict
    for name, info in students.items():
        avg = calculate_average(info["grades"])
        letter = get_letter_graade(avg)
        remark = get_grade_remark(letter)
        print(f" {name:<20} {info['subject']:<15} {avg:>6.2f} {letter:>6}  {remark}")

def class_statistics():
    """ Shjow overall class performance statistics. """
    if not students:
        print(" No Students Found")
        return
    
    # List Comprehension: build a flat list of all averages
    all_averages = [calculate_average(info["grades"]) for info in students.values()]

    # Statistics module for advance math 
    mean_avg   = statistics.mean(all_averages)
    median_avg = statistics.median(all_averages)
    stdev_avg  = statistics.stdev(all_averages) if len(all_averages) > 1 else 0

    # Sorted by average --- sorted() returns a New list, doesn't modify original 
    ranked = sorted(students.items(), 
                    key=lambda item: calculate_average(item[1]["grades"]),
                    reverse=True)          # lambda: anonymous function used as sort key
    
    topper_name, __ = ranked[0]
    lowest_name, __ = ranked[-1]

    print(f"""
        Class Statistics
        -------------------------------------------------
        | Total Students : {len(students)}
        | Class Average  : {mean_avg:.2f}
        | Median Average : {median_avg:.2f}
        | Std Deviation  : {stdev_avg:.2f}
        | Top Performer  : {topper_name}
        | Needs Support  : {lowest_name}
        --------------------------------------------------""")
    
    print("\n Class Ranking.")
    for rank, (name, info) in enumerate(ranked, 1):
        avg = calculate_average(info["grades"])
        letter = get_letter_graade(avg)
        print(f" {rank}. {name:<20} {avg:.2f} ({letter})")

def grade_distribution():
    """Show how many students fall in each grade  band. """
    if not students:
        print(" No Students Found .")
        return
    distribution = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}

    for info in students.values():        # .values() iterates over dict values
        avg = calculate_average(info["grades"])
        letter = get_letter_graade(avg)
        distribution[letter] += 1
    
    print("\n GRADE DISTRIBUTION!")
    for letter, count in distribution.items():
        bar = " " * count         #visual bar chart using string repetition
        print(f" {letter} | {bar:<20} {count} students(s)")

#-------------------------------------------------------
# SECTION 5: File I/O  - Save & Load CSV
# Concept: csv module, file open/close, reading rows
#-------------------------------------------------------

CSV_FILE = "students.csv"

def save_to_csv():
    """Save all student data to a CSV file. """
    if not students:
        print(" Nothing to Save")
        return
    # open() with 'w' mode - creates or overwrites file 
    with open(CSV_FILE, "w", newline="") as file:  # "with" ensures file is closed safely
        writer = csv.writer(file)
        writer.writerow(["Name", "Subject", "Grades"])  # header row

        for name, info in students.items():
            grades_str = ";".join(str(g) for g in info["grades"])  #join list --> string
            writer.writerow([name, info["subject"], grades_str])

    print(f" Data Saved to '{CSV_FILE}'.")

def load_from_csv():
    """ Load Student data from a CSV file into memory. """
    if not os.path.exists(CSV_FILE):
        print(f" '{CSV_FILE}' not found.")
        return
    
    students.clear()    # clear exisiting data before laoding 

    with open(CSV_FILE, "r") as file:
        reader = csv.DictReader(file)          # DictReader maps each row to a dict using haeders
        for row in reader:                     # each row is a dict: {"name": ...., "Grades": ....}
            name = row["Name"]
            subject = row["Subject"]
            grades = [float(g) for g in row["Grades"].split(";")]   # split string  -- list of floats
            students[name] = {"student": subject, "grades": grades}
    
    print(f" Data Loaded from '{CSV_FILE}'. {len(students)} student(s) found.")

#-----------------------------------------------------
# SECTION 6: Input Helpers
# Concepts: Input validation, type casting , loops
#-----------------------------------------------------

def get_grades_input():
    """"Prompt user to enter multiple grades separed by commas , """
    while True:
        raw = input(" Enter grades (comma separeted, e.g: 85, 90, 76): ").strip()
        try:
            grades = [float(g.strip()) for g in raw.split(",")]   # list comprehension + split
            if all(0 <= g <= 100 for g in grades):                # all() checks every item
                return grades
            else:
                print(" Grades must be between 0 and 100 ")
        except ValueError:
            print(" Invalid input. Use numbers separated by commans")

def get_name_input(prompt=" Enter student Name: "):
    """Get a non-empty name fro the user. """
    while True:
        name = input(prompt).strip()
        if name:
            return name
        print(" Name cannot be empty.")

#------------------------------------------------------------
# SECTION 7: Main Menu Loop
# Concept: while loop, multi-level menu, string comparision 
#------------------------------------------------------------

def main():
    print("\n" + "="*45)
    print(" STUDENT GRADE MANAGER !!!!")
    print("="*45)

    # Load existing data on startup if CSV exists
    if os.path.exists(CSV_FILE):
        load_from_csv()
    while True:
        print("""
        -------------- MAIN MENU ----------------
              [0] Exit
              [1] Add Student 
              [2] Update Student Grades
              [3] Delete Student
              [4] View Student Report
              [5] View All Students
              [6] Class Statistics & Ranking
              [7] Grade Distribution
              [8] Save to CSV
              [9] Load from CSV
              [0] Exit
----------------------------------------------------""")
        choice = input(" Enter choice: ").strip()

        if choice == "1":
            name    = get_name_input( " Student Name : ")
            subject = input(" Subject  : ").strip() or "General"
            grades  = get_name_input()
            add_student(name, subject, grades)

        elif choice == "2":
            name = get_grades_input()
            new_grades = get_grades_input()
            update_grades(name, new_grades)

        elif choice == "3":
            name = get_name_input()
            confirm = nput(f" Delete '{name.title()}' ? (Yes/No): ").strip().lower()
            if confirm == "yes":
                delete_student(name)
            else:
                print(" Cancelled ")
        
        elif choice == "4":
            name = get_grades_input()
            view_student(name)

        elif choice == "5":
            view_all_students()
        
        elif choice == "6":
            class_statistics()
        
        elif choice == "7":
            grade_distribution()
        
        elif choice == "8":
            save_to_csv()
        
        elif choice == "9":
            load_from_csv()
        
        elif choice == "0":
            save_to_csv()
            print("\n Data Saved. GoodBye!! \n")
            break
        else:
            print(" Invalid Choice... Try again! ")

#-----------------------------------------------
# Entry Point 
#-----------------------------------------------

if __name__ == "__main__":
    main()
