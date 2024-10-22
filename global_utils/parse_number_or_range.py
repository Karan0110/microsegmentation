def parse_number_or_range(value : str) -> range:
    try:
        if '-' in value:
            # This is a range like 32-329
            start, end = map(int, value.split('-'))
            if start > end:
                raise ValueError(f"Invalid range: '{value}' (start should be less than or equal to end)")
            return range(start, end + 1)
        else:
            # This is a single number like 23
            num = int(value)
            return range(num, num + 1)
    except ValueError:
        raise ValueError(f"Invalid input format: '{value}' (must be a number or a range in format START-END)")
