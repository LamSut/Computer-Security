while True:
    try:
        upper_bound = 300.0
        lower_bound = 0.0
        height = float(input("Enter your height in centimeters: "))
        if height < lower_bound or height > upper_bound:
            print("Height must be between", lower_bound, "and", upper_bound)
            continue
        weight = float(input("Enter your weight in kilograms: "))
        if weight < 0:
            print("Weight cannot be negative.")
            continue
        bmi = weight / ((height / 100) ** 2)
        if bmi < 18.5:
            print("You are underweight.")
        elif bmi >= 18.5 and bmi <= 24.9:
            print("You are normal weight.")
        elif bmi >= 25 and bmi <= 29.9:
            print("You are overweight.")
        else:
            print("You are obese.")
        break
    except ValueError:
        print("Invalid input. Please try again.")