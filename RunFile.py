import main


def calc_range(n):
    lower_bound = (n // 10) * 10
    upper_bound = lower_bound + 10
    return f"{lower_bound}-{upper_bound}"


result, mae = main.mainFunction('archive_2/FGNET/images/014A40.JPG')

print(f"Age estimation between {calc_range(result)} Years")
print(f"Mean absolute Error is: {mae}")
