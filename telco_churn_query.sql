SELECT 
	customer_id, churn, dependents, device_protection, gender, monthly_charges, multiple_lines, paperless_billing, partner,  phone_service,tenure, online_backup, 		online_security, senior_citizen, streaming_tv, streaming_movies, tech_support, total_charges,
	i.internet_service_type_id AS 'internet_service_type_id', internet_service_type,
	ct.contract_type_id AS 'contract_type_id', contract_type,
	p.payment_type_id AS 'payment_type_id', payment_type
FROM customers AS c
JOIN contract_types AS ct ON ct.`contract_type_id` = c.contract_type_id
JOIN internet_service_types AS i ON i.internet_service_type_id = c.internet_service_type_id
JOIN payment_types AS p ON p.payment_type_id = c.payment_type_id;