# Manual Calculations for Part 3: Gradient Descent

This directory contains handwritten manual calculations for Part 3 of Formative 3.

## Required File

**File to add:** `Part 3 - Manual Calculation of Gradient Descent.pdf`

## Instructions for Creating the PDF

### What to Include

Your manual calculations should demonstrate a complete gradient descent iteration with the following steps:

1. **Initial Setup**
   - Write down the initial parameter values (θ₀, θ₁)
   - State the learning rate (α)
   - List your training data points

2. **Forward Pass (Predictions)**
   - Calculate predictions ŷ for each data point using: ŷ = θ₀ + θ₁x
   - Show all arithmetic clearly

3. **Cost Function Calculation**
   - Compute the Mean Squared Error (MSE) cost
   - Show the calculation: J(θ) = (1/2m) Σ(ŷᵢ - yᵢ)²

4. **Gradient Calculations (Chain Rule)**
   - Calculate ∂J/∂θ₀ (gradient with respect to intercept)
   - Calculate ∂J/∂θ₁ (gradient with respect to slope)
   - **Show the chain rule steps explicitly**
   - Include partial derivatives for each term

5. **Parameter Updates**
   - Update θ₀: θ₀_new = θ₀_old - α(∂J/∂θ₀)
   - Update θ₁: θ₁_new = θ₁_old - α(∂J/∂θ₁)
   - Show all arithmetic

6. **Verification**
   - Optionally show a second iteration to demonstrate understanding

### Formatting Guidelines

- **Write neatly** - calculations should be legible
- **Label each section** clearly (Predictions, Cost, Gradients, Updates)
- **Show all intermediate steps** - don't skip arithmetic
- **Use proper mathematical notation**
- **Box or highlight final answers** for each section

### Creating the PDF

1. Write your calculations by hand on paper
2. Scan the pages or take clear photographs
   - Ensure good lighting and contrast
   - Make sure all writing is readable
3. Combine all pages into a single PDF file
4. Name the file: `Part3_Manual_Gradient_Descent.pdf`
5. Place it in this directory (`Formative3/manual_calculations/`)

## Team Collaboration

- **All team members** should review and understand these calculations
- **Lead:** Chinemerem is responsible for coordinating this part
- Discuss the calculations as a team to ensure everyone understands the gradient descent process

## Why This Matters

Manual calculations help you:
- Understand what happens "under the hood" in gradient descent
- Verify that your code implementation is correct
- Build intuition for how learning rate and gradients affect convergence
- Prepare for exam questions that may ask for manual calculations

---

**Note:** This manual work complements the Python implementation in the Jupyter notebook. Make sure your hand calculations match the logic in your code!
