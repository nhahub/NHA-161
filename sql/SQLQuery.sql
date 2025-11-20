
-- Customer Churn SQL Analysis Script
-- ----------------------------------
-- This script creates the database schema, loads data, and provides queries for churn analysis.

-- 1. Create Database and Tables
CREATE DATABASE CustomerChurnDB;
GO
USE CustomerChurnDB;
GO

-- Customers table
CREATE TABLE Customers (
    CustomerID INT PRIMARY KEY,
    CustomerName NVARCHAR(100),
    Age INT,
    Gender NVARCHAR(10)
);

-- Purchases table
CREATE TABLE Purchases (
    PurchaseID INT IDENTITY(1,1) PRIMARY KEY,
    CustomerID INT,
    PurchaseDate DATETIME,
    ProductCategory NVARCHAR(50),
    ProductPrice DECIMAL(10,2),
    Quantity INT,
    TotalPurchaseAmount DECIMAL(10,2),
    PaymentMethod NVARCHAR(50),
    Returns NVARCHAR(20),
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);

-- Customer status table
CREATE TABLE CustomerStatus (
    CustomerID INT PRIMARY KEY,
    ChurnStatus NVARCHAR(20),
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);

-- 2. Churn Statistics
-- Churn distribution
SELECT 
    ChurnStatus,
    COUNT(*) AS CustomerCount,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM CustomerStatus), 2) AS Percentage
FROM CustomerStatus
GROUP BY ChurnStatus;

-- Average spending by churn status
SELECT 
    cs.ChurnStatus,
    AVG(p.TotalPurchaseAmount) AS AvgSpending
FROM Purchases p
JOIN CustomerStatus cs ON p.CustomerID = cs.CustomerID
GROUP BY cs.ChurnStatus;

-- Most purchased product categories by churned customers
SELECT 
    p.ProductCategory,
    COUNT(*) AS TotalPurchases
FROM Purchases p
JOIN CustomerStatus cs ON p.CustomerID = cs.CustomerID
WHERE cs.ChurnStatus = 'churned'
GROUP BY p.ProductCategory
ORDER BY TotalPurchases DESC;

-- Payment method analysis by churn status
SELECT 
    p.PaymentMethod,
    cs.ChurnStatus,
    COUNT(*) AS Count
FROM Purchases p
JOIN CustomerStatus cs ON p.CustomerID = cs.CustomerID
GROUP BY p.PaymentMethod, cs.ChurnStatus
ORDER BY Count DESC;

-- 3. Data Import from Excel (if needed)
-- Enable Ad Hoc Distributed Queries (run as admin if needed)
EXEC sp_configure 'show advanced options', 1;
RECONFIGURE;
EXEC sp_configure 'Ad Hoc Distributed Queries', 1;
RECONFIGURE;
GO

-- Import data from Excel (update path as needed)
SELECT *
INTO TempCustomerData
FROM OPENROWSET(
    'Microsoft.ACE.OLEDB.12.0',
    'Excel 12.0;Database=C:\\Data\\data_large.xlsx;HDR=YES',
    'SELECT * FROM [Sheet 1$]'
);
GO

-- Insert data into tables
INSERT INTO Customers (CustomerID, CustomerName, Age, Gender)
SELECT DISTINCT [Customer#ID], [Customer#Name], [Age], [Gender]
FROM TempCustomerData;

INSERT INTO Purchases (CustomerID, PurchaseDate, ProductCategory, ProductPrice, Quantity,
                       TotalPurchaseAmount, PaymentMethod, Returns)
SELECT [Customer#ID], [Purchase#Date], [Product#Category], [Product#Price],
       [Quantity], [Total#Purchase#Amount], [Payment#Method], [Returns]
FROM TempCustomerData;

INSERT INTO CustomerStatus (CustomerID, ChurnStatus)
SELECT DISTINCT [Customer#ID], [Churn]
FROM TempCustomerData;
GO

-- 4. Customer Churn Report Table (optional, for BI/reporting)
CREATE TABLE CustomerChurnReport (
    CustomerID INT PRIMARY KEY,
    CustomerName NVARCHAR(100),
    Gender NVARCHAR(10),
    Age INT,
    JoinDate DATE,
    LastPurchaseDate DATE,
    TotalOrders INT,
    TotalSpent DECIMAL(12,2),
    AverageSpentPerOrder DECIMAL(12,2),
    PreferredPaymentMethod NVARCHAR(50),
    TotalReturns INT,
    ReturnRate NVARCHAR(10),
    MostPurchasedCategory NVARCHAR(50),
    ChurnPredicted NVARCHAR(5),
    ChurnProbabilityScore DECIMAL(4,2),
    Notes NVARCHAR(255)
);

-- 5. Customer Summary Report (Individual Analysis)
SELECT 
    c.CustomerID,
    c.CustomerName,
    c.Gender,
    c.Age,
    COUNT(p.PurchaseID) AS TotalOrders,
    SUM(p.TotalPurchaseAmount) AS TotalSpent,
    AVG(p.TotalPurchaseAmount) AS AvgSpentPerOrder,
    MAX(p.PurchaseDate) AS LastPurchaseDate,
    SUM(CASE WHEN p.Returns = 'Return' THEN 1 ELSE 0 END) AS TotalReturns,
    ROUND(100.0 * SUM(CASE WHEN p.Returns = 'Return' THEN 1 ELSE 0 END) / COUNT(p.PurchaseID), 2) AS ReturnRate,
    cs.ChurnStatus AS PredictedChurnStatus
FROM Customers c
LEFT JOIN Purchases p ON c.CustomerID = p.CustomerID
LEFT JOIN CustomerStatus cs ON c.CustomerID = cs.CustomerID
GROUP BY c.CustomerID, c.CustomerName, c.Gender, c.Age, cs.ChurnStatus
ORDER BY TotalSpent DESC;

