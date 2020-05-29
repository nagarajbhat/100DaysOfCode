# OOPs concept - PYTHON
# most of the examples here has been taken from turotials by Corey schafer
# PART 1 - CREATE A CLASS
# We can create a class using 
class Employee:
    pass

e1 = Employee()
e1.salary = 25000
print("\npart 1")
#.__dict__ method gives dictionary values for that instance
print(e1.__dict__)
print(e1)

# PART 2 - INITIALIZE VARIABLES, CLASS VARIABLES

class Employee:
    #class variable - class variable applies for the whole class
    raise_amt = 1.2
    #Initialize variable - This is basically a constructor
    def __init__(self,first_name,last_name,salary):
        self.first_name = first_name
        self.last_name = last_name
        self.email = first_name +"."+ last_name + "@company.com"
        self.salary = salary

e2 = Employee("Aron","Paul",70000)
print("\npart 2")

print("\n instance variables : ",e2.__dict__)
print("class variables: ",Employee.__dict__)


#PART 3 - CLASS METHODS AND STATIC METHODS
#You can create a class method by using @classmethod decorater

class Employee:
     #class variable
    raise_amt = 1.2
    #Initialize variable
    def __init__(self,first_name,last_name,salary):
        self.first_name = first_name
        self.last_name = last_name
        self.email = first_name +"."+ last_name + "@company.com"
        self.salary = salary
    
    def raise_salary(self):
        self.salary = self.salary*self.raise_amt

    #class method
    @classmethod
    def raise_amount(cls,raise_amt):
        cls.raise_amt = raise_amt
    
    
e3 = Employee("karl","Paul",80000)
print("\npart 3")
print(e3.__dict__)
e3.raise_amount(1.5)
e3.raise_salary()

print(e3.__dict__)

# PART 4 -  INHERITENCE

class Developer(Employee):
    
    def __init__(self,first_name,last_name,salary,prog_lang):
        super().__init__(first_name,last_name,salary)
        self.prog_lang = prog_lang

print("\npart 4")
emp4 = Employee("joey","tribianny",57000)
dev1 = Developer("chandler","bing",80000,"python")
print(emp4.__dict__)
print(dev1.__dict__)
