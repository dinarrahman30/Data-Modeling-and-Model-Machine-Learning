--==== rata-rata umur customer jika dilihat dari marital statusnya

select "MaritalStatus", round(avg("Age"),2) as rata_rata_umur
from (
	select c."Age", c."MaritalStatus" 
	from public."transaction" t
	left join public.customers c on t."CustomerID" = c."CustomerID" 
	) as tb
group by "MaritalStatus"

--==== rata-rata umur customer jika dilihat dari gender

select "Gender", round(avg("Age"),2) as rata_rata_umur
from (
	select c."Age", c."Gender" 
	from public."transaction" t
	left join public.customers c on t."CustomerID" = c."CustomerID" 
	) as tb
group by "Gender"

--==== nama store dengan total quantity terbanyak

select "StoreName", sum(t."qty") as total
from store s 
inner join "transaction" t on t."StoreID" = s."StoreID" 
group by "StoreName" 
order by total desc 
limit 1;

--==== nama produk terlaris dengan total amount terbanyak

select p."ProductName", sum(t."TotalAmount") as total
from product p 
inner join "transaction" t on p."ProductID" = t."ProductID" 
group by p."ProductName" 
order by total desc 
limit 1;