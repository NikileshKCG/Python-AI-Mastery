"""
Project 3: OOP Bank Account System
Concepts : Classes, Objects, Inheritance, Encapsulation, Polymorphism,
        Class Methods, Static Methods, Properties, Exception Handling,
        Dunder Methods, File I/O (JSON)
"""

import json
import os
import datetime

#----------------------------------------------------------------
# Section 1: Custom Exceptions
# Concepts : Creating your own exception classes via inheritance
#----------------------------------------------------------------

class InsufficientFundsError(Exception):
    """Raised when withdrawl exceeds available balance. """
    pass

class InvalidAmountError(Exception):
    """Raised when a transaction amount is zero or negative."""
    pass

class AccountLockedError(Exception):
    """Raised when a locked account is accessed. """
    pass

#-----------------------------------------------------------
# Section 2: Base Class - BankAccount
# Concepts : Calls definitions, __init__, instance variables,
#           encapsulation (private vars wtih __), properties,
#           dunder methods (__str__, __repr__)
#-----------------------------------------------------------

class BankAccount:
    """
    Base class representing a generic bank account.
    Encapsulates balance and transaction history.
    """

    bank_name = "PyBank"    # Class variable - shared across ALL instances
    total_accounts = 0      # Class variable - tacks how many accounts exist

    def __init__(self, owner, account_number, initial_balance=0.0):
        # Instance variables -- unique to each object
        self.owner          = owner.strip().title()
        self.account_number = account_number
        self.account_type   = "Basic"
        self.__balance      = float(initial_balance)    #__ makes it PRIVATE (encapsulation)
        self.__transactions = []                        # private transaction logs
        self.__is_locked    = False

        BankAccount.total_accounts += 1

        # log the opening transaction 
        self._record_transaction("Account Opened", initial_balance)
    
    #-------------- Properties -----------------
    # Concept: @property lets you access a method like an attribute
    # It protects private data while allowing read access

    @property
    def balance(self):
        """ Read-only access to private balance. """
        return self.__balance
    
    @property
    def is_locked(self):
        """ Read-only access to lock status. """
        return self.__is_locked
    
    @property
    def transactions(self):
        """ Return a copy of transactions - prevents direct modifications. """
        return self.__transactions.copy()
    
    # -------- Protected Helper ----------
    # Concept: Singe underscore _ = "protected" - usable by subclasses

    def _record_transaction(self, transaction_type, amount, note=""):
        """ Log a transaction with timestamp. """
        entry = {
            "type"      : transaction_type,
            "amount"    : amount,
            "balance"   : self.__balance,
            "timestamp" : datetime.datetime.now().strftime("%Y-%m-%D %H:%M:%S"),
            "note"      : note
        }
        self.__transactions.append(entry)

    def _validate_amount(self, amount):
        """ Raise error if the amount is not positive. """
        if amount <= 0:
            raise InvalidAmountError(f"Amount must be postivie. Got: {amount}")
    
    def _check_lock(self):
        """Raise error if account is locked. """
        if self.__is_locked:
            raise AccountLockedError(f" Account {self.account_number} is locked. ")
        
    # ------- Core Methonds -----------------
    
    def deposit(self, amount, note=""):
        """ Deposit Money into the account. """
        self._check_lock()
        self._validate_amount(amount)
        self.__balance += amount
        self._record_transaction("Deposit", amount, note)
        print(f" Deposited ${amount:,.2f} | New Balance: ${self.__balance:,.2f}")

    def withdraw(self, amount, note=""):
        """ Withdraw money from the account. """
        self._check_lock()
        self._validate_amount(amount)
        if amount > self.__balance:
            raise InsufficientFundsError(
                f" Cannot withdraw ${amount:,.2f}. Balance: ${self.__balance:,.2f}"
            )
        self.__balance -= amount
        self._record_transaction("Withdrawal", amount, note)
        print(f" Withdrawn ${amount:,.2f} | New Balance: ${self.__balance:,.2f}")
    
    def transfer(self, amount, target_account, note=""):
        """ Transfer money to another account. """
        self._check_lock()
        self._validate_amount(amount)
        self.withdraw(amount, note=f"Transfer to {target_account.account_number}")
        target_account.deposit(amount, note=f"Transfer from {self.account_number}")
        print(f" Traferred ${amount:,.2f} to {target_account.owner}")

    def lock_account(self):
        self.__is_locked = True
        print(f" Accpunt {self.account_number} is locked !! ")
    
    def unlock_account(self):
        self.is_locked = False
        print(f" Account {self.account_number} is unlocked !!")
    
    def get_statement(self):
        """ Print a formatted Account Statement. """
        print(f"""
================================================================
              {self.bank_name} - Account Statement
    =======================================================
        Owner   : {self.owner:<32} ||
        Acc No  : {self.account_number:<32} ||
        Type    : {self.account_type:<32} ||
        Balance : ${self.__balance:<31,.2f} ||
        Locked  : {'Yes' if self.__is_locked else 'No':<32} ||
================================================================""")
        
        print(f"\n {'#':<4} {'Type':<15'} {'Amount':>10} {'Balance':>12} {'Date':<20}")
        print(" " + "-" * 65)
        for i, t in enumerate(self.__transactions, 1):
            print(f" {i:<4} {t['type']:<15} ${t['amount']:>9,.2f} ${t['balance']:>11,.2f} {t["timestamp"]}")
    #---------- Class Method & Staic Method -------------
    # Concept: classmethod works on the call itself (not instance)
    #           staticmethod is a utility - doesn't need self or cls

    @classmethod
    def get_total_accounts(cls):
        """Return total number of accounts created """
        return cls.total_accounts
    @staticmethod
    def validate_account_number(acc_no):
        """ check if account number format is valid (e.g: ACC12334). """
        return isinstance(acc_no, str) and acc_no.startswith("ACC") and len(acc_no) == 7
    
    #--------------- Dunder (MAGIC) Method------------------
    # Concept: Special methods Python calls automatically 

    def __str__(self):
        """Human-readable string - used by print() """
        return f"{self.bank_name} | {self.owner} | {self.account_number} | ${self.__balance:,.2f}"
    
    def __repr__(self):
        """Developer-readable string -- used in debugging """
        return f"BankAccount(owner='{self.owner}', acc='{self.account_number}', balance={self.__balance})"
    
    def __eq__(self, other):
        """Check if two account have the same account number. """
        return self.account_number == other.account_number
    
    def __lt__(self, other):
        """Compare accounts by balance - enables sorting """
        return self.balance < other.balance
    
    def to_dict(self):
        """ Serialize account data for JSON storage. """
        return {
            "owner"          : self.owner,
            "account_number" : self.account_number,
            "account_type"   : self.account_type,
            "balance"        : self.__balance,
            "is_locked"      : self.__is_locked,
            "transactions"   : self.__transactions
        }

#-------------------------------------------------------------
# Section 3: Inheritance - SavingsAccount 
# Concept: Child class inherits from patent,
#           super(), method overriding
#-------------------------------------------------------------

class SavingsAccount(BankAccount):
    """ 
    Saving account with interest and withdrawl limits. 
    Inherits all BankAccount functionality.
    """
    INTEREST_RATE = 0.04        # 4% Annual interest - class variable
    MAX_WITHDRAWLS = 3          # limit per month

    def __init__(self, owner, account_number, initial_balance=0.0):
        super().__init__(owner, account_number, initial_balance)    #call parent __init__
        self.account_type       = "Savings"
        self.__withdrawal_count = 0             # tracks withdrawal per month

    def withdraw(self, amount, note=""):
        """ Override parent withdraw - enforce withdrawal limit"""
        if self.__withdrawal_count >= SavingsAccount.MAX_WITHDRAWLS:
            raise Exception(
                f"Monthly withdrawal limit ({SavingsAccount.MAX_WITHDRAWLS}) reached. "
            )
        super().withdraw(amount, note)
        self.__withdrawal_count += 1
        remaining = SavingsAccount.MAX_WITHDRAWLS - self.__withdrawal_count
        print(f" Withdrawals reamining this Month: {remaining}")
    
    def apply_interest(self):
        """ Add Monthly interest to balance """
        interest = round(self.balance * (self.INTEREST_RATE / 12), 2)
        self.deposit(interest, note="Monthly Interest")
        print(f" Interest applied: ${interest:,.2f} at {self.INTEREST_RATE*100}% p.a ")
    
    def reset_monthly_withdrawals(self):
        """Reset withdrawal counter (call at start of each month)."""
        self.__withdrawal_count = 0
        print(" Monthly withdrawal count reset!")

#-----------------------------------------------------------
# Section 4: Inheritance - CurrentAccount
# Concepts: Multiple Inheritance levels, overdraft feature
#-----------------------------------------------------------

class CurrentAccount(BankAccount):
    """ 
    Current (checking) account with overdraft facility.
    """

    def __init__(self, owner, account_number, initial_balance=0.0, overdraft_limit=500.0):
        super().__init__(owner, account_number, initial_balance)
        self.account_type    = "Current"
        self.overdraft_limit = overdraft_limit      #extra borrowing limit
    
    def withdraw(self, amount, note=""):
        """Override -- allow withdrawal into overdraft. """
        self._check_lock()
        self._validate_amount(amount)
        available = self.balance + self.overdraft_limit
        if amount > available:
            raise InsufficientFundsError(
                f"Exceeds overdraft limit. Available ${available:,.2f}"
            )
        # Call grandparent logic manually since we skip parent's balnce check
        # We use BankAccount's deposit/withdraw by adjusting balance via parent's withdraw
        # Here we directly call _BankAccount__balance workaround via parent withdraw 
        # Best Practice: call parent and catch, or restructure validation in base
        super().withdraw(amount, note)
    
    def get_overdraft_status(self):
        used = max(0, -self.balance)
        print(f" Overdraft Used : ${used:,.2f}")
        print(f" Overdraft Limit: ${self.overdraft_limit:,.2f}")
        print(f" Remaining      : ${self.overdraft_limit - used:,.2f}")

#------------------------------------------------------------
# Section 5: Bank - Container Class
# Concept: A class that manages a collection of objects
#           Demonstrates composition (Bank has accounts)
#------------------------------------------------------------

class Bank:
    """Manages all accounts in the Bank. """
    DATA_FILE = "bank_data.json"

    def __init__(self):
        self.accounts = {}  # dict: account_number  --> account object
    
    def _next_account_number(self):
        """ Auto-Generate next account number like ACC1001, ACC1002, ACC1003...."""
        num = 1001 + len(self.accounts)
        return f"ACC{num}"
    
    def create_account(self, owner, account_type="savings", initial_balance=0.0):
        """Create a new account and add to bank . """
        acc_no = self._next_account_number()

        if account_type.lower() == "savings":
            account = SavingsAccount(owner, acc_no, initial_balance)
        elif account_type.lower() == "current":
            account = CurrentAccount(owner, acc_no, initial_balance)
        else:
            account = BankAccount(owner, acc_no, initial_balance)
        
        self.accounts[acc_no] = account
        print(f" Account created! Number: {acc_no} | Type: {account.account_type}")
        return account
    
    def get_account(self, acc_no):
        """ Retrieve account details by number. """
        acc_no = acc_no.strip().upper()
        if acc_no not in self.accounts:
            print(f" Account '{acc_no} not found. ")
            return None
        return self.accounts[acc_no]
    
    def list_all_accounts(self):
        """ Dispaly summary of all accounts """
        if not self.accounts:
            print(" No Accounts Found !")
            return
        print(f"\n {'Acc No':<10} {'Owner':<20} {'Type':<12} {'Balance':>12} {'Locked'}")
        print(" " + "-" * 62)
        for acc in sorted(self.accounts.values(), reverse=True):  #uses __lt__
            locked = "Locked" if acc.is_locked else " "
            print(f" {acc.account_number:<10} {acc.owner:<20} {acc.account_type:<12} ${acc.balance:>11,.2f} {locked}")
    
    def total_deposits(self):
        """ Returns total money held acroos all accounts. """
        total = sum(acc.balance for acc in self.account.values())   # generator expression
        print(f"\n Total funds in {BankAccount.bank_name}: ${total:,.2f}")
        return total
    
    def save_data(self):
        """Persist all accounts to JSON file. """
        data = {acc_no: acc.to_dict() for acc_no, acc in self.accounts.items()}
        with open(self.DATA_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        print(f" DATA saved to '{self.DATA_FILE}'.")
    
    def load_data(self):
        """ Load accounts from JSON file """
        if not os.path.exists(self.DATA_FILE):
            return
        with open(self.DATA_FILE, "r") as f:
            data = json.load(f)
        for acc_no, info in data.tems():
            atype = info.get("account_type", "Basic")
            if atype == "Savings":
                acc = SavingsAccount(info["owner"], acc_no, info["balance"])
            elif atype == "Current":
                acc = CurrentAccount(info["owner"], acc_no, info["balance"])
            else:
                acc = BankAccount(info["owner"], acc_no, info["balance"])
            self.account[acc_no] = acc
        print(f" Loaded {len(self.accounts)} account(s) from '{self.DATA_FILE}'.")

#--------------------------------------------------------------
# Section 6: Main Menu
#--------------------------------------------------------------

def get_amount(prompt="Enter Amount: S"):
    while True:
        try:
            val = float(input(prompt).strip())
            return val
        except ValueError:
            print(" Enter a valid number ")

def main():
    print("\n" + "="*45)
    print(f" Welcome to {BankAccount.bank_name}")
    print("="*45)

    bank = Bank()
    bank.load_data()

    while True:
        print("""
------------------ MAIN MENU ----------------------------------------
              [1] Create Account
              [2] Deposit
              [3] Withdraw
              [4] Trasfer
              [5] View Statement
              [6] List All Accounts
              [7] Apply Interest (Savings)
              [8] Lock /Unlock Account
              [9] Bank total funds
              [10] Save Data
              [0] Exit
--------------------------------------------------------------------""")
        
        choice = input(" Enter choice: ").strip()
        try:
            if choice == "1":
                owner = input(" Owner Name : ").strip()
                print(" Account type - [1] Savings [2] Current [3] Basic")
                t = input("Choose : ").strip()
                atype = {"1": "savings", "2": "current", "3":"basic"}.get(t, "savings")
                bal = get_amount("Initial deposit: $")
                bank.create_account(owner, atype, bal)

            elif choice == "2":
                acc_no = input(" Account number: ").strip()
                acc = bank.get_account(acc_no)
                if acc:
                    amt = get_amount()
                    note = input(" Note (Optional): ").strip()
                    acc.deposit(amt, note)
            
            elif choice == "3":
                acc_no = input(" Account number: ").strip()
                acc = bank.get_account(acc_no)
                if acc:
                    amt = get_amount()
                    note = input(" Note (Optional): ").strip()
                    acc.withdraw(amt, note)
            
            elif choice == "4":
                from_no = input(" From account: ").strip()
                to_no = input(" To account : ").strip()
                src = bank.get_account(from_no)
                dst = bank.get_account(to_no)
                if src and dst:
                    amt = get_amount()
                    src.transfer(amt, dst)
            
            elif choice == "5":
                acc_no = input("Account number: ").strip()
                acc = bank.get_account(acc_no)
                if acc:
                    acc.get_statement()
            
            elif choice == "6":
                bank.list_all_accounts()
            
            elif choice == "7":
                acc_no = input("Savings Account Number: ").strip()
                acc = bank.get_account(acc_no)
                if acc and isinstance(acc, SavingsAccount):   #instance checks class type
                    acc.apply_interest()
                elif acc:
                    print(" Interest only applies to Savings Accounts.")
            
            elif choice == "8":
                acc_no = input("Account Number: ").strip()
                acc = bank.get_account(acc_no)
                if acc:
                    action = input(" [1] Lock [2] Unlock :").strip()
                    if action == "1":
                        acc.lock_account()
                    elif action == "2":
                        acc.unlock.account()
            
            elif choice == "9":
                bank.total_deposits()
            
            elif choice == "10":
                bank.save_data()

            elif choice == "0":
                bank.save_data()
                print(f"\n Thank you for banking with {BankAccount.bank_name}!\n ")
                break

            else:
                print(" Invalid Choice ")

        # Cathing custom and built-in exceptions        
        except InsufficientFundsError as e:
            print(f" Insufficient Funds: {e}")
        except InvalidAmountError as e:
            print(f" Invalid Amount: {e}")
        except AccountLockedError as e:
            print(f" Account Locked: {e}")
        except Exception as e:
            print(f" Error: {e}")

#----------------------------------------------------------
# Entry Point
#----------------------------------------------------------

if __name__ == "__main__":
    main()