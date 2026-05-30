import random
import pandas as pd


def generate_dataset(
    num_customers=20,
    grid_size=100,
    min_demand=1,
    max_demand=10,
    seed=42,
    output_file="data/customers.csv"
):
    random.seed(seed)

    customers = []

    for customer_id in range(1, num_customers + 1):

        x = random.randint(0, grid_size)
        y = random.randint(0, grid_size)

        demand = round(random.uniform(min_demand, max_demand), 2)

        customers.append({
            "id": customer_id,
            "x": x,
            "y": y,
            "demand": demand
        })

    df = pd.DataFrame(customers)

    df.to_csv(output_file, index=False)

    print(f"Dataset saved to {output_file}")
    print(df.head())


if __name__ == "__main__":
    generate_dataset()