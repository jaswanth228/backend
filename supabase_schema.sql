-- OPTIONAL. The backend creates these tables automatically on startup (SQLAlchemy create_all).
-- Use this only to inspect/lock down things from the Supabase SQL editor.
-- Defence in depth: the backend uses the service_role key (bypasses RLS); enabling RLS blocks anyone using the public anon key.
DO $$
DECLARE t text;
BEGIN
  FOR t IN SELECT tablename FROM pg_tables WHERE schemaname='public' LOOP
    EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', t);
  END LOOP;
END $$;
-- Run AFTER the first server start (so the tables exist). The 'patient-images' bucket is private by default.
