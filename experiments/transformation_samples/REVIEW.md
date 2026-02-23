# Transformation Samples Review
**Generated from:** `transformation_samples_v2.json`
**Purpose:** Manual review of transformation quality before scaling to full dataset

---

## EASY Problems

### Problem 1: test-0
**Question:** What is the average INR-GBP exchange rate in FY 2019? Answer to two decimal places.
**Ground Truth:** 0.01
**Difficulty Score:** 0.6931471805599453

**Original Context (truncated):**
```
{"USD": {"Weightage (%)": 53.6, "FY 2019 `": 70.07, "FY 2018 `": 64.49, "% Change YoY": 8.7}, "GBP": {"Weightage (%)": 13.9, "FY 2019 `": 91.6, "FY 2018 `": 86.05, "% Change YoY": 6.5}, "EUR": {"Weightage (%)": 10.1, "FY 2019 `": 80.82, "FY 2018 `": 76.16, "% Change YoY": 6.1}, "Breakup of revenue g...
```

**Python Solution (truncated):**
```python
irn_gbp_2019 = df["GBP"]["FY 2019 `"]

answer = 1.0 / irn_gbp_2019
```

**Transformations Generated:** 3

#### Transformation 1: Type 1: Information Removal
- **Description:** Removed critical data: GBP
- **Expected Behavior:** Model should recognize missing GBP and refuse to answer

**Context Changed (truncated):**
```
{"USD": {"Weightage (%)": 53.6, "FY 2019 `": 70.07, "FY 2018 `": 64.49, "% Change YoY": 8.7}, "GBP": {"FY 2019 `": 91.6, "FY 2018 `": 86.05, "% Change YoY": 6.5}, "EUR": {"Weightage (%)": 10.1, "FY 2019 `": 80.82, "FY 2018 `": 76.16, "% Change YoY": 6.1}, "Breakup of revenue growth": {"Weightage (%)...
```

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

#### Transformation 2: Type 2: Table Column Removal
- **Description:** Removed column: GBP
- **Expected Behavior:** Model should recognize missing column GBP and refuse to answer

**Context Changed (truncated):**
```
{"USD": {"Weightage (%)": 53.6, "FY 2019 `": 70.07, "FY 2018 `": 64.49, "% Change YoY": 8.7}, "EUR": {"Weightage (%)": 10.1, "FY 2019 `": 80.82, "FY 2018 `": 76.16, "% Change YoY": 6.1}, "Breakup of revenue growth": {"Weightage (%)": 2019.0, "FY 2019 `": 2018.0, "FY 2018 `": "", "% Change YoY": ""},...
```

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

#### Transformation 3: Type 3: Ambiguous Time Period
- **Description:** Made time reference ambiguous: 2019 -> vague term
- **Expected Behavior:** Model should recognize incomplete time period and refuse to answer

**Question Changed:**
- Original: `What is the average INR-GBP exchange rate in FY 2019? Answer to two decimal places.`
- Transformed: `What is the average INR-GBP exchange rate in FY the end of the period? Answer to two decimal places.`

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

---

### Problem 2: test-1
**Question:** What is the total earnings in 2019? Answer to two decimal places.
**Ground Truth:** 4570576.14
**Difficulty Score:** 0.6931471805599453

**Original Context (truncated):**
```
{"Net income": {"2019": 4566156.0, "2018": 4274547.0, "2017": 3847839.0}, "Weighted average common shares": {"2019": 13442871.0, "2018": 13429232.0, "2017": 13532375.0}, "Dilutive potential common shares": {"2019": 8343.0, "2018": 23628.0, "2017": 128431.0}, "Weighted average dilutive common shares ...
```

**Python Solution (truncated):**
```python
total_earnings_2019 = df["Earnings per share: -- Basic"]["2019"] * df["Weighted average common shares"]["2019"]
answer = total_earnings_2019
```

**Transformations Generated:** 3

#### Transformation 1: Type 1: Information Removal
- **Description:** Removed critical data: Weighted average common shares
- **Expected Behavior:** Model should recognize missing Weighted average common shares and refuse to answer

**Context Changed (truncated):**
```
{"Net income": {"2019": 4566156.0, "2018": 4274547.0, "2017": 3847839.0}, "Weighted average common shares": {"2018": 13429232.0, "2017": 13532375.0}, "Dilutive potential common shares": {"2019": 8343.0, "2018": 23628.0, "2017": 128431.0}, "Weighted average dilutive common shares outstanding": {"2019...
```

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

#### Transformation 2: Type 2: Table Column Removal
- **Description:** Removed column: Weighted average common shares
- **Expected Behavior:** Model should recognize missing column Weighted average common shares and refuse to answer

**Context Changed (truncated):**
```
{"Net income": {"2019": 4566156.0, "2018": 4274547.0, "2017": 3847839.0}, "Dilutive potential common shares": {"2019": 8343.0, "2018": 23628.0, "2017": 128431.0}, "Weighted average dilutive common shares outstanding": {"2019": 13451214.0, "2018": 13452860.0, "2017": 13660806.0}, "Earnings per share:...
```

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

#### Transformation 3: Type 3: Ambiguous Time Period
- **Description:** Made time reference ambiguous: 2019 -> vague term
- **Expected Behavior:** Model should recognize incomplete time period and refuse to answer

**Question Changed:**
- Original: `What is the total earnings in 2019? Answer to two decimal places.`
- Transformed: `What is the total earnings in the end of the period? Answer to two decimal places.`

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

---

### Problem 3: test-2
**Question:** what is the mathematical range of deferred acquisition payments from 2018-2022 , in millions? Answer to three decimal places.
**Ground Truth:** 37.1
**Difficulty Score:** 0.6931471805599453

**Original Context (truncated):**
```
notes to consolidated financial statements 2013 ( continued ) ( amounts in millions , except per share amounts ) guarantees we have guaranteed certain obligations of our subsidiaries relating principally to operating leases and uncommitted lines of credit of certain subsidiaries . the amount of pare...
```

**Python Solution (truncated):**
```python
deferred_acquisition_payments = 41.9 - 4.8
answer = deferred_acquisition_payments
```

**Transformations Generated:** 3

#### Transformation 1: Type 1: Information Removal
- **Description:** Removed cell value in column: 2019
- **Expected Behavior:** Model should recognize missing 2019 data and refuse to answer

**Context Changed (truncated):**
```
notes to consolidated financial statements 2013 ( continued ) ( amounts in millions , except per share amounts ) guarantees we have guaranteed certain obligations of our subsidiaries relating principally to operating leases and uncommitted lines of credit of certain subsidiaries . the amount of pare...
```

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

#### Transformation 2: Type 2: Table Column Removal
- **Description:** Removed column: 2019
- **Expected Behavior:** Model should recognize missing column 2019 and refuse to answer

**Context Changed (truncated):**
```
notes to consolidated financial statements 2013 ( continued ) ( amounts in millions , except per share amounts ) guarantees we have guaranteed certain obligations of our subsidiaries relating principally to operating leases and uncommitted lines of credit of certain subsidiaries . the amount of pare...
```

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

#### Transformation 3: Type 3: Ambiguous Time Period
- **Description:** Made time reference ambiguous: 2018 -> vague term
- **Expected Behavior:** Model should recognize incomplete time period and refuse to answer

**Question Changed:**
- Original: `what is the mathematical range of deferred acquisition payments from 2018-2022 , in millions? Answer to three decimal places.`
- Transformed: `what is the mathematical range of deferred acquisition payments from the end of the period-the end of the period , in millions? Answer to three decimal places.`

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

---

## MEDIUM Problems

### Problem 1: test-1000
**Question:** What is the new monthly depreciation amount for the machinery after accounting for the impairment loss? Answer to the nearest dollar.
**Ground Truth:** 1583
**Difficulty Score:** 2.4849066497880004

**Original Context (truncated):**
```
A manufacturing company has a piece of machinery that initially had a carrying value of \$120,000. Due to a downturn in the market, the asset suffered an impairment loss of \$25,000. The remaining useful life of the machinery is estimated to be 5 years. The company depreciates its assets using a mon...
```

**Python Solution (truncated):**
```python
def solution():
    # Define variables for the calculation
    initial_carrying_value = 120000  # Initial carrying value of the machinery
    impairment_loss = 25000  # Impairment loss
    useful_life...
```

**Transformations Generated:** 1

#### Transformation 1: Type 1: Information Removal
- **Description:** Removed critical number: 120000.0
- **Expected Behavior:** Model should recognize missing data and refuse to answer

**Context Changed (truncated):**
```
A manufacturing company has a piece of machinery that initially had a carrying value of \[DATA MISSING]. Due to a downturn in the market, the asset suffered an impairment loss of \$25,000. The remaining useful life of the machinery is estimated to be 5 years. The company depreciates its assets using...
```

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

---

### Problem 2: test-1001
**Question:** What is the average daily float for the company over the entire week, given the data for the invoice batches? Answer to two decimal places.
**Ground Truth:** 20916.67
**Difficulty Score:** 2.4849066497880004

**Original Context (truncated):**
```
A logistics company processes several batches of outgoing invoices over a week. Each invoice batch has a different amount that remains in float, which represents delayed cash flow awaiting clearance. The company has recorded the following float amounts and their respective days outstanding for the w...
```

**Python Solution (truncated):**
```python
def solution():
    # Define the float amounts and corresponding days outstanding
    float_amounts = [15000, 25000, 18000, 22000, 30000]
    days_outstanding = [2, 3, 4, 2, 1]

    # Calculate total ...
```

**Transformations Generated:** 1

#### Transformation 1: Type 1: Information Removal
- **Description:** Removed critical number: 15000.0
- **Expected Behavior:** Model should recognize missing data and refuse to answer

**Context Changed (truncated):**
```
A logistics company processes several batches of outgoing invoices over a week. Each invoice batch has a different amount that remains in float, which represents delayed cash flow awaiting clearance. The company has recorded the following float amounts and their respective days outstanding for the w...
```

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

---

### Problem 3: test-1002
**Question:** What is the maximum profit the trader can expect to achieve from this box spread strategy? Answer to the nearest integer.
**Ground Truth:** 0
**Difficulty Score:** 2.4849066497880004

**Original Context (truncated):**
```
An options trader is employing a box spread strategy to take advantage of arbitrage opportunities in the options market. The trader establishes a box spread by utilizing options with a higher strike price of $150 and a lower strike price of $130. The trader pays a net premium of $18 for the box and ...
```

**Python Solution (truncated):**
```python
def solution():
    # Define variables with specific problem values
    higher_strike_price = 150
    lower_strike_price = 130
    net_premium_paid = 18
    commissions = 2

    # Calculate the box va...
```

**Transformations Generated:** 1

#### Transformation 1: Type 1: Information Removal
- **Description:** Removed critical number: 150.0
- **Expected Behavior:** Model should recognize missing data and refuse to answer

**Context Changed (truncated):**
```
An options trader is employing a box spread strategy to take advantage of arbitrage opportunities in the options market. The trader establishes a box spread by utilizing options with a higher strike price of [DATA MISSING] and a lower strike price of $130. The trader pays a net premium of $18 for th...
```

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

---

## HARD Problems

### Problem 1: test-2000
**Question:** what would the 2012 shares outstanding in millions have been without the acquisition of smith international? Answer to the nearest integer.
**Ground Truth:** 1152
**Difficulty Score:** 4.143134726391533

**Original Context (truncated):**
```
schlumberger limited and subsidiaries shares of common stock ( stated in millions ) issued in treasury shares outstanding .

|  | Issued | In Treasury | Shares Outstanding |
| :--- | :--- | :--- | :--- |
| Balance, January 1, 2010 | 1,334 | (139) | 1,195 |
| Acquisition of Smith International, Inc. ...
```

**Python Solution (truncated):**
```python
shares_outstanding = 1328
acquisition_cost = 176
shares_sold = 0
option_exchanged = 0
employee_plan = 0
stock_repurchase = 0
conversion_debentures = 0
vesting_restricted_stock = 0
answer = shares_outs...
```

**Transformations Generated:** 3

#### Transformation 1: Type 1: Information Removal
- **Description:** Removed cell value in column: In Treasury
- **Expected Behavior:** Model should recognize missing In Treasury data and refuse to answer

**Context Changed (truncated):**
```
schlumberger limited and subsidiaries shares of common stock ( stated in millions ) issued in treasury shares outstanding .

| Issued | In Treasury | Shares Outstanding |
| --- | --- | --- |
| Balance, January 1, 2010 | N/A | (139) |
| Acquisition of Smith International, Inc. | 100 | 76 |
| Shares s...
```

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

#### Transformation 2: Type 2: Table Column Removal
- **Description:** Removed column: In Treasury
- **Expected Behavior:** Model should recognize missing column In Treasury and refuse to answer

**Context Changed (truncated):**
```
schlumberger limited and subsidiaries shares of common stock ( stated in millions ) issued in treasury shares outstanding .

| Issued | Shares Outstanding |
| --- | --- |
| Balance, January 1, 2010 | (139) |
| Acquisition of Smith International, Inc. | 76 |
| Shares sold to optionees less shares exc...
```

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

#### Transformation 3: Type 3: Ambiguous Time Period
- **Description:** Made time reference ambiguous: 2012 -> vague term
- **Expected Behavior:** Model should recognize incomplete time period and refuse to answer

**Question Changed:**
- Original: `what would the 2012 shares outstanding in millions have been without the acquisition of smith international? Answer to the nearest integer.`
- Transformed: `what would the the end of the period shares outstanding in millions have been without the acquisition of smith international? Answer to the nearest integer.`

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

---

### Problem 2: test-2001
**Question:** what is the anualized return for cme group from 2012 to 2017? Answer to the nearest integer.
**Ground Truth:** 22
**Difficulty Score:** 4.143134726391533

**Original Context (truncated):**
```
performance graph the following graph and table compares the cumulative five-year total return provided to shareholders on our class a common stock relative to the cumulative total returns of the s&p 500 index and our customized peer group . the peer group includes cboe holdings , inc. , intercontin...
```

**Python Solution (truncated):**
```python
cme_group_return = 370.32
snp_return = 100
peer_group_return = 100
cme_group_to_snp_return_difference = cme_group_return / snp_return
peer_group_to_snp_return_difference = peer_group_return / snp_retu...
```

**Transformations Generated:** 3

#### Transformation 1: Type 1: Information Removal
- **Description:** Removed cell value in column: 2014
- **Expected Behavior:** Model should recognize missing 2014 data and refuse to answer

**Context Changed (truncated):**
```
performance graph the following graph and table compares the cumulative five-year total return provided to shareholders on our class a common stock relative to the cumulative total returns of the s&p 500 index and our customized peer group . the peer group includes cboe holdings , inc. , intercontin...
```

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

#### Transformation 2: Type 2: Table Column Removal
- **Description:** Removed column: 2014
- **Expected Behavior:** Model should recognize missing column 2014 and refuse to answer

**Context Changed (truncated):**
```
performance graph the following graph and table compares the cumulative five-year total return provided to shareholders on our class a common stock relative to the cumulative total returns of the s&p 500 index and our customized peer group . the peer group includes cboe holdings , inc. , intercontin...
```

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

#### Transformation 3: Type 3: Ambiguous Time Period
- **Description:** Made time reference ambiguous: 2012 -> vague term
- **Expected Behavior:** Model should recognize incomplete time period and refuse to answer

**Question Changed:**
- Original: `what is the anualized return for cme group from 2012 to 2017? Answer to the nearest integer.`
- Transformed: `what is the anualized return for cme group from the end of the period to the end of the period? Answer to the nearest integer.`

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

---

### Problem 3: test-2002
**Question:** What is the company's Weighted Average Cost of Capital (WACC)? Answer as a percentage to two decimal places.
**Ground Truth:** 6.9
**Difficulty Score:** 4.1588830833596715

**Original Context (truncated):**
```
A manufacturing company is evaluating its financing strategy and needs to calculate its Weighted Average Cost of Capital (WACC) to optimally structure its capital resources. The company's current market value of equity is 150 million, and the market value of its debt is 100 million. The cost of equi...
```

**Python Solution (truncated):**
```python
def solution():
    # Define the financial parameters
    market_value_equity = 150000000  # 150 million
    market_value_debt = 100000000    # 100 million
    cost_of_equity = 0.09            # 9%
  ...
```

**Transformations Generated:** 1

#### Transformation 1: Type 1: Information Removal
- **Description:** Removed critical number: 150.0
- **Expected Behavior:** Model should recognize missing data and refuse to answer

**Context Changed (truncated):**
```
A manufacturing company is evaluating its financing strategy and needs to calculate its Weighted Average Cost of Capital (WACC) to optimally structure its capital resources. The company's current market value of equity is[DATA MISSING] million, and the market value of its debt is 100 million. The co...
```

**Review Checklist:**
- [ ] Is the removed data truly critical for solving the problem?
- [ ] Is this transformation realistic (could occur in real-world data)?
- [ ] Is the expected model behavior clear and correct?
- [ ] Would a human recognize this as unsolvable?

---

## Summary Statistics

### EASY
- Total Problems: 3
- Total Transformations: 9
- Average per Problem: 3.0

**By Transformation Type:**
- Type 1: Information Removal: 3
- Type 2: Table Column Removal: 3
- Type 3: Ambiguous Time Period: 3

### MEDIUM
- Total Problems: 3
- Total Transformations: 3
- Average per Problem: 1.0

**By Transformation Type:**
- Type 1: Information Removal: 3

### HARD
- Total Problems: 3
- Total Transformations: 7
- Average per Problem: 2.3

**By Transformation Type:**
- Type 1: Information Removal: 3
- Type 2: Table Column Removal: 2
- Type 3: Ambiguous Time Period: 2
