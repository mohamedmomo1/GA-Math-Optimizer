import random
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GeneticAlgorithmOptimizer:
    def __init__(self, master):
        self.master = master
        master.title("Multi-Function Genetic Algorithm Optimizer")
        master.geometry("1200x900")

        parameters_frame = ttk.LabelFrame(master, text="Optimization Parameters")
        parameters_frame.pack(padx=10, pady=10, fill=tk.X)

        ttk.Label(parameters_frame, text="Select Optimization Function:").grid(row=0, column=0, padx=5, pady=5)
        self.function_var = tk.StringVar()
        self.function_dropdown = ttk.Combobox(parameters_frame, 
            textvariable=self.function_var, 
            values=[
                "Sphere Function", 
                "Sum of Squares Function", 
                "Booth Function", 
                "Beale Function", 
                "Matyas Function"
            ])
        self.function_dropdown.grid(row=0, column=1, padx=5, pady=5)
        self.function_dropdown.set("Sphere Function")
        self.function_dropdown.bind("<<ComboboxSelected>>", self.update_parameter_bounds)

        settings = [
            ("Population Size:", "pop_size", "50"),
            ("Generations:", "generations", "100"),
            ("Dimension/Chromosome Size:", "chromosome_size", "5"),
            ("Lower Bound:", "lower_bound", "-5"),
            ("Upper Bound:", "upper_bound", "5"),
            ("Mutation Rate:", "mutation_rate", "0.1")
        ]

        self.entries = {}
        for i, (label, key, default) in enumerate(settings, start=1):
            ttk.Label(parameters_frame, text=label).grid(row=i, column=0, padx=5, pady=5)
            entry = ttk.Entry(parameters_frame)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[key] = entry

        self.run_button = ttk.Button(parameters_frame, text="Run Optimization", command=self.run_genetic_algorithm)
        self.run_button.grid(row=len(settings)+1, column=0, columnspan=2, padx=5, pady=5)

        self.result_frame = ttk.LabelFrame(master, text="Results")
        self.result_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.figure, self.fitness_ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.result_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.best_solution_label = ttk.Label(self.result_frame, text="", wraplength=600)
        self.best_solution_label.pack(padx=5, pady=5)

    def update_parameter_bounds(self, event=None):
        function = self.function_var.get()
        
        if function == "Sphere Function":
            self.entries['lower_bound'].delete(0, tk.END)
            self.entries['lower_bound'].insert(0, "-5")
            self.entries['upper_bound'].delete(0, tk.END)
            self.entries['upper_bound'].insert(0, "5")
            self.entries['chromosome_size'].delete(0, tk.END)
            self.entries['chromosome_size'].insert(0, "5")
        
        elif function == "Sum of Squares Function":
            self.entries['lower_bound'].delete(0, tk.END)
            self.entries['lower_bound'].insert(0, "-10")
            self.entries['upper_bound'].delete(0, tk.END)
            self.entries['upper_bound'].insert(0, "10")
            self.entries['chromosome_size'].delete(0, tk.END)
            self.entries['chromosome_size'].insert(0, "5")
        
        elif function in ["Booth Function", "Beale Function", "Matyas Function"]:
            self.entries['lower_bound'].delete(0, tk.END)
            self.entries['chromosome_size'].delete(0, tk.END)
            self.entries['chromosome_size'].insert(0, "2")
            
            if function == "Booth Function":
                self.entries['lower_bound'].insert(0, "-10")
                self.entries['upper_bound'].delete(0, tk.END)
                self.entries['upper_bound'].insert(0, "10")
            
            elif function == "Beale Function":
                self.entries['lower_bound'].insert(0, "-5")
                self.entries['upper_bound'].delete(0, tk.END)
                self.entries['upper_bound'].insert(0, "5")
            
            elif function == "Matyas Function":
                self.entries['lower_bound'].insert(0, "-10")
                self.entries['upper_bound'].delete(0, tk.END)
                self.entries['upper_bound'].insert(0, "10")

    def objective_function(self, x):
        function = self.function_var.get()
        
        if function == "Sphere Function":
            return sum([xi ** 2 for xi in x])
        
        elif function == "Sum of Squares Function":
            return sum((i + 1) * x[i]**2 for i in range(len(x)))
        
        elif function == "Booth Function":
            return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
        
        elif function == "Beale Function":
            return (1.5 - x[0] + x[0]*x[1])**2 + \
                   (2.25 - x[0] + x[0]*x[1]**2)**2 + \
                   (2.625 - x[0] + x[0]*x[1]**3)**2
        
        elif function == "Matyas Function":
            return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

    def initialize_population(self, pop_size, chromosome_size, lower_bound, upper_bound):
        return [[random.uniform(lower_bound, upper_bound) for _ in range(chromosome_size)] for _ in range(pop_size)]

    def evaluate_population(self, population):
        return [self.objective_function(individual) for individual in population]

    def select_parents(self, population, fitness):
        num_parents = max(2, int(len(population) * 0.1))
        
        population_fitness = list(zip(population, fitness))
        population_fitness.sort(key=lambda x: x[1])
        
        parents = [ind for ind, _ in population_fitness[:num_parents]]
        
        return parents

    def crossover(self, parent1, parent2):
        alpha = random.random()
        child = [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(parent1, parent2)]
        return child

    def mutate(self, individual, mutation_rate, lower_bound, upper_bound):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] += random.uniform(-1, 1)
                individual[i] = max(lower_bound, min(upper_bound, individual[i]))
        return individual

    def run_genetic_algorithm(self):
        try:
            pop_size = int(self.entries['pop_size'].get())
            generations = int(self.entries['generations'].get())
            chromosome_size = int(self.entries['chromosome_size'].get())
            lower_bound = float(self.entries['lower_bound'].get())
            upper_bound = float(self.entries['upper_bound'].get())
            mutation_rate = float(self.entries['mutation_rate'].get())

            function = self.function_var.get()
            if function in ["Booth Function", "Beale Function", "Matyas Function"] and chromosome_size != 2:
                messagebox.showwarning("Warning", f"{function} requires exactly 2 dimensions. Resetting to 2.")
                chromosome_size = 2
                self.entries['chromosome_size'].delete(0, tk.END)
                self.entries['chromosome_size'].insert(0, "2")

            population = self.initialize_population(pop_size, chromosome_size, lower_bound, upper_bound)

            best_fitness_history = []

            best_solution = min(population, key=self.objective_function)
            best_fitness = self.objective_function(best_solution)

            for generation in range(generations):
                fitness = self.evaluate_population(population)

                current_best = min(population, key=self.objective_function)
                current_best_fitness = self.objective_function(current_best)

                if current_best_fitness < best_fitness:
                    best_solution = current_best
                    best_fitness = current_best_fitness

                best_fitness_history.append(best_fitness)

                parents = self.select_parents(population, fitness)

                next_generation = []
                while len(next_generation) < pop_size - 1:
                    parent1, parent2 = random.sample(parents, 2)
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child, mutation_rate, lower_bound, upper_bound)
                    next_generation.append(child)

                next_generation.append(best_solution)
                population = next_generation

            self.fitness_ax.clear()

            self.fitness_ax.plot(range(1, len(best_fitness_history) + 1), best_fitness_history, color='blue')
            self.fitness_ax.set_title(f'{function} - Fitness Improvement')
            self.fitness_ax.set_xlabel('Generation Number')
            self.fitness_ax.set_ylabel('Best Fitness Value')
            self.fitness_ax.grid(True, linestyle='--', alpha=0.7)

            self.canvas.draw()

            if function in ["Booth Function", "Beale Function", "Matyas Function"]:
                solution_text = f"Best Solution: x = {best_solution[0]:.4f}, y = {best_solution[1]:.4f}\nBest Fitness: {best_fitness:.6f}"
            else:
                solution_text = f"Best Solution: {best_solution}\nBest Fitness: {best_fitness:.6f}"
            
            self.best_solution_label.config(text=solution_text)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values")

def main():
    root = tk.Tk()
    app = GeneticAlgorithmOptimizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()